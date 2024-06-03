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

EXP_ID=${3:-"pix2pix"}
MODEL_NAME="stabilityai/stable-diffusion-2"
ANN_PATH=ANNOTATION_PATH

$CMD train_ldm_goal_image.py \
    --resolution=256 --random_flip --train_batch_size=32 --seed=42 \
    --gradient_accumulation_steps=1 --gradient_checkpointing \
    --max_train_steps=180000 --checkpointing_steps=1000 --checkpoints_total_limit=30 \
    --learning_rate=5e-05 --max_grad_norm=1 --lr_warmup_steps=0 --conditioning_dropout_prob=0.05 \
    --mixed_precision=fp16 --resume_from_checkpoint "latest" \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --output_dir "./lavis/output/LDM/runs/$EXP_ID" \
    --ann_path=$ANN_PATH \
    # --include_depth