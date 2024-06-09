NUM_GPUS_PER_NODE=${1:-1}
NUM_NODES=${2:-1}

echo "NUM_GPUS_PER_NODE: $NUM_GPUS_PER_NODE"
echo "NUM_NODES: $NUM_NODES"

if [ $NUM_NODES -gt 1 ]; then
    ip=YOUR_MASTER_IP
    NODE_RANK=YOUR_NODE_RANK
    CMD="torchrun --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$ip --node_rank=$NODE_RANK --rdzv_backend static"
else
    MASTER_PORT=$(($$%1000+24000))
    CMD="torchrun --nproc_per_node=$NUM_GPUS_PER_NODE --master_port=$MASTER_PORT"
fi

$CMD train_pe_goal_pcd.py