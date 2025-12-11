# export HCCL_OP_EXPANSION_MODE="AIV"
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=2

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
# export LCCL_PARALLEL=1


# FLUX_DIR=/home/weights/FLUX.1-dev/ torchrun --nproc_per_node=2 --master_port=12345 run_flux_cp_npu.py --attn "_mindie_sd_la" --height 960 --width 1920 --steps 2 --parallel ulysses --vae-dp

WAN_2_2_DIR=/home/weights/Wan2.1-T2V-14B-Diffusers/ torchrun --nproc_per_node=8 --master-port 12345 run_wan_cp_npu.py --attn "_mindie_sd_la" --height 1024 --width 1024 --steps 10 --parallel ulysses --vae-dp
#  WAN_2_2_DIR=/home/weights/Wan2.1-I2V-14B-720P-Diffusers/ torchrun --nproc_per_node=8 --master-port 12345 run_wan_i2v_cp_npu.py --attn "_native_npu" --height 1024 --width 1024 --steps 10 --parallel ulysses --vae-dp
