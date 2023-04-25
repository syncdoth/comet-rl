ids=$1
N_GPUS=$2

script="main.py \
    --model_name comet_pretrained/comet-atomic_2020_BART \
    --log_with wandb \
    --learning_rate 2e-5 \
    --batch_size 16 --mini_batch_size 4 --gradient_accumulation_steps 1 \
    --num_epochs 5 --max_step -1 --save_step 250 --eval_step 250 \
    --model_save_path checkpoints/comet-atomic_2020_BART-ppo \
    --train_data_path ckbp_data/train.csv \
    --evaluation_data_path ckbp_data/ckbp2.0.csv \
    --tracker_project_name comet-ppo \
    --exp_name 'bart-default-param' \
    --add_rel_token"


if [ $N_GPUS = 1 ]; then
    echo "Using 1 GPU: use simple python launcher..."
    script="CUDA_VISIBLE_DEVICES=$ids python $script"
else
    echo "Using multi-GPU: using torchrun launcher..."
    script="CUDA_VISIBLE_DEVICES=$ids WORLD_SIZE=$N_GPUS torchrun --nproc_per_node $N_GPUS $script"
fi

eval $script
