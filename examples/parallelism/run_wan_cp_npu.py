import os
import sys

sys.path.append("..")

import time

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.distributed as dist

from diffusers import WanPipeline, WanTransformer3DModel
from diffusers.utils import export_to_video
from utils import (
    cachify,
    get_args,
    maybe_destroy_distributed,
    maybe_init_distributed,
    strify,
)

import cache_dit
from cache_dit.npu_optim import npu_optimize


def init_profiler(step=1):
    experimental_config = torch_npu.profiler._ExperimentalConfig(
    	export_type=torch_npu.profiler.ExportType.Text,
    	profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    	msprof_tx=False,
    	aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
    	l2_cache=False,
    	op_attr=False,
    	data_simplification=False,
    	record_op_args=False,
    	gc_detect_threshold=None
    )

    prof = torch_npu.profiler.profile(
    	activities=[
    		torch_npu.profiler.ProfilerActivity.CPU,
    		torch_npu.profiler.ProfilerActivity.NPU
    		],
    	schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=step, repeat=1, skip_first=0),
    	on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
    	record_shapes=True,
    	profile_memory=False,
    	with_stack=False,
    	with_modules=False,
    	with_flops=False,
    	experimental_config=experimental_config)
    return prof


def run_pipe(args, pipe, warmup: bool = False, prof=None):
    prompt = "A cat walks on the grass, realistic"
    negative_prompt = \
        "Bright tones, overexposed, static, blurred details, " \
        "subtitles, style, works, paintings, images, static, overall gray, " \
        "worst quality, low quality, JPEG compression residue, ugly, incomplete, " \
        "extra fingers, poorly drawn hands, poorly drawn faces, deformed, " \
        "disfigured, misshapen limbs, fused fingers, still picture, messy " \
        "background, three legs, many people in the background, walking backwards"

    seed = 1234
    generator = torch.Generator(device="cpu").manual_seed(seed)

    num_inference_steps = args.steps if not warmup else 3
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=49,
        guidance_scale=5.0,
        generator=generator,
        num_inference_steps=num_inference_steps,
        prof=prof,
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

    pipe = WanPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )

    if args.cache or args.parallel_type is not None:
        cachify(args, pipe)

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

    pipe.set_progress_bar_config(disable=rank != 0)

    # warmup
    _ = run_pipe(args, pipe, warmup=True)
    
    prof = None
    # prof = True
    if prof:
        prof = init_profiler(args.steps + 2)

    start = time.time()
    video = run_pipe(args, pipe, prof=prof)
    end = time.time()

    if rank == 0:
        cache_dit.summary(pipe)

        time_cost = end - start
        save_path = f"wan.{strify(args, pipe)}.mp4"
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
