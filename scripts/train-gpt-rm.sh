ids=$1
arrIDs=(${ids//,/ })
N_GPUS="${#arrIDs[@]}"

script="main.py \
    --model_name comet_pretrained/gpt2xl-comet-atomic-2020 \
    --log_with wandb \
    --learning_rate 2e-5 \
    --batch_size 16 --mini_batch_size 4 --gradient_accumulation_steps 1 \
    --num_epochs 5 --max_step -1 --save_step 250 --eval_step 250 \
    --model_save_path checkpoints/gpt2xl-comet-atomic-2020-ppo \
    --train_data_path ckbp_data/train.csv \
    --evaluation_data_path ckbp_data/ckbp2.0.csv \
    --tracker_project_name comet-ppo \
    --exp_name 'gpt2-pseudoreasoner-rm' \
    --add_rel_token \
    --use_rm \
    --reward_model_ptlm roberta-large \
    --reward_model_saved_path checkpoints/reward_model-PseudoReasoner/best_model_seed_100.pth"


if [ $N_GPUS = 1 ]; then
    echo "Using 1 GPU: use simple python launcher..."
    script="CUDA_VISIBLE_DEVICES=$ids python $script"
else
    echo "Using multi-GPU: using torchrun launcher..."
    script="CUDA_VISIBLE_DEVICES=$ids WORLD_SIZE=$N_GPUS torchrun --nproc_per_node $N_GPUS $script"
fi

eval $script
