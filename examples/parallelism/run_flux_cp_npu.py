import os
import sys

sys.path.append("..")

import time
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

from diffusers import (
    FluxPipeline,
    FluxTransformer2DModel,
    PipelineQuantizationConfig,
)
from utils import (
    get_args,
    strify,
    cachify,
    maybe_init_distributed,
    maybe_destroy_distributed,
)
import cache_dit
from cache_dit.npu_optim import npu_optimize


npu_optimize([
    "npu_fast_gelu",
    "npu_rms_norm",
    "npu_layer_norm_eval",
    "npu_rotary_mul",
    "npu_weight_nz",
    "npu_adalayernorm",
    "flux_transformer_block_forward"
])

args = get_args()
print(args)

rank, device = maybe_init_distributed(args)
world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

pipe: FluxPipeline = FluxPipeline.from_pretrained(
    os.environ.get(
        "FLUX_DIR",
        "black-forest-labs/FLUX.1-dev",
    ),
    torch_dtype=torch.bfloat16,
).to("cuda")

if args.cache or args.parallel_type is not None:
    cachify(args, pipe)

if args.vae_dp:
    HW_SPLITS = {
        1: (1, 1),
        2: (1, 2),
        4: (2, 2),
        8: (2, 4),
    }
    pipe.vae.enable_dp(world_size=world_size, hw_splits=HW_SPLITS[world_size]) # , overlap_ratio=0.01, overlap_pixels=64)

assert isinstance(pipe.transformer, FluxTransformer2DModel)

pipe.set_progress_bar_config(disable=rank != 0)

def init_profiler(steps):
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
    	schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=steps, repeat=1, skip_first=0),
    	on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
    	record_shapes=True,
    	profile_memory=False,
    	with_stack=False,
    	with_modules=False,
    	with_flops=False,
    	experimental_config=experimental_config)
    return prof


def run_pipe(pipe: FluxPipeline, prof=None):
    image = pipe(
        "A cat holding a sign that says hello world",
        height=1024 if args.height is None else args.height,
        width=1024 if args.width is None else args.width,
        num_inference_steps=28 if args.steps is None else args.steps,
        generator=torch.Generator("cpu").manual_seed(0),
        prof=prof,
    ).images[0]
    return image


if args.compile:
    cache_dit.set_compile_configs()
    pipe.transformer = torch.compile(pipe.transformer)


# warmup
_ = run_pipe(pipe)

prof = None
# prof = True
if prof:
    prof = init_profiler(steps=args.steps+2)

start = time.time()

image = run_pipe(pipe, prof)

end = time.time()

if rank == 0:
    cache_dit.summary(pipe)

    time_cost = end - start
    save_path = f"flux.{strify(args, pipe)}.png"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving image to {save_path}")
    image.save(save_path)

maybe_destroy_distributed()
