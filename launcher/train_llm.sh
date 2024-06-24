NUM_GPUS_PER_NODE=${1:-1}
NUM_NODES=${2:-1}

echo "NUM_GPUS_PER_NODE: $NUM_GPUS_PER_NODE"
echo "NUM_NODES: $NUM_NODES"

if [ $NUM_NODES -gt 1 ]; then
    ip=YOUR_MASTER_IP
    NODE_RANK=YOUR_NODE_RANK
    CMD="torchrun --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$ip --node_rank=$NODE_RANK"
else
    MASTER_PORT=$(($$%1000+24000))
    CMD="torchrun --nproc_per_node=$NUM_GPUS_PER_NODE --master_port=$MASTER_PORT"
fi

# ======== CHANGE HERE ========
JOB_ID=${3:-"Lang"}
CFG_PATH="lavis/projects/blip2/train/pretrain/3d_flant5_robo_pretrain.yaml"
# ======== CHANGE HERE ========

$CMD train.py --cfg-path $CFG_PATH --job_id $JOB_ID
