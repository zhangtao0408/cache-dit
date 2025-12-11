import os
import sys

sys.path.append("..")

import time

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import numpy as np
import torch.distributed as dist

from diffusers import WanImageToVideoPipeline, WanTransformer3DModel
from diffusers.utils import export_to_video, load_image
from utils import (
    cachify,
    get_args,
    maybe_destroy_distributed,
    maybe_init_distributed,
    strify,
)

import cache_dit
from cache_dit.npu_optim import npu_optimize


def run_pipe(args, pipe, image, warmup: bool = False):
    # prompt = "A cat walks on the grass, realistic"
    # negative_prompt = "Bright tones, overexposed, static, blurred details, "
    "subtitles, style, works, paintings, images, static, overall gray, "
    "worst quality, low quality, JPEG compression residue, ugly, incomplete, "
    "extra fingers, poorly drawn hands, poorly drawn faces, deformed, "
    "disfigured, misshapen limbs, fused fingers, still picture, messy "
    "background, three legs, many people in the background, walking backwards"
    prompt = (
        "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in "
        "the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
    )
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    seed = 1234
    generator = torch.Generator(device="cpu").manual_seed(seed)

    num_inference_steps = args.steps if not warmup else 3
    output = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=image.height,
        width=image.width,
        num_frames=49,
        guidance_scale=5.0,
        generator=generator,
        num_inference_steps=num_inference_steps,
    ).frames[0]
    return output


def main():
    args = get_args()
    print(args)

    rank, device = maybe_init_distributed(args)
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    model_id = os.environ.get(
        "WAN_2_2_DIR",
        # "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    )

    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    ) 

    if args.cache or args.parallel_type is not None:
        from cache_dit import ParallelismConfig
        from cache_dit.parallelism.parallel_backend import ParallelismBackend
        from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput

        backend = (
            ParallelismBackend.NATIVE_PYTORCH
            if args.parallel_type in ["tp"]
            else ParallelismBackend.NATIVE_DIFFUSER
        )

        parallel_kwargs = (
            {
                "attention_backend": (
                    "_native_cudnn" if not args.attn else args.attn
                ),
                "cp_plan": {
                    "rope": {
                        0: ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
                        1: ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
                    },
                    "blocks.0": {
                        "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
                    },
                    "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
                }
            }
            if backend == ParallelismBackend.NATIVE_DIFFUSER
            else None
        )

        parallelism_config=(
            ParallelismConfig(
                ulysses_size=(
                    dist.get_world_size()
                    if args.parallel_type == "ulysses"
                    else None
                ),
                ring_size=(
                    dist.get_world_size()
                    if args.parallel_type == "ring"
                    else None
                ),
                tp_size=(
                    dist.get_world_size()
                    if args.parallel_type == "tp"
                    else None
                ),
                backend=backend,
                parallel_kwargs=parallel_kwargs,
            )
        )

        cachify(args, pipe, parallelism_config=parallelism_config)

    if args.cpu_offload:
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)

    if args.vae_dp:
        HW_SPLITS = {
            1: (1, 1),
            2: (1, 2),
            4: (2, 2),
            8: (2, 4),
        }
        pipe.vae.enable_dp(world_size=world_size, hw_splits=HW_SPLITS[world_size]) # , overlap_ratio=0.01, overlap_pixels=64)

    if args.vae_tiling:
        pipe.vae.enable_tiling(
            # tile_sample_min_height=int(args.height / 2 * 3),
            # tile_sample_min_width=int(args.width / 2 * 3),
            # tile_sample_stride_height=int(args.height / 2),
            # tile_sample_stride_width=int(args.width / 2),
        )

    assert isinstance(pipe.transformer, WanTransformer3DModel)

    image = load_image("/home/y30061107/vips/infer/astronaut.jpg")

    # max_area = args.height * args.width
    # aspect_ratio = image.height / image.width
    # mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1] * world_size
    # height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    # width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    height = args.height
    width = args.width

    image = image.resize((width, height))


    pipe.set_progress_bar_config(disable=rank != 0)

    # warmup
    _ = run_pipe(args, pipe, image, warmup=True)

    start = time.time()
    video = run_pipe(args, pipe, image)
    end = time.time()

    if rank == 0:
        cache_dit.summary(pipe)

        time_cost = end - start
        save_path = f"wan_i2v.{strify(args, pipe)}.mp4"
        print(f"Time cost: {time_cost:.2f}s")
        print(f"Saving image to {save_path}")
        export_to_video(video, save_path, fps=16)

    maybe_destroy_distributed()


if __name__ == "__main__":
    npu_optimize([
        "npu_fast_gelu",
        "npu_rms_norm",
        "npu_layer_norm_eval",
        "npu_rotary_mul",
    ])
    main()
