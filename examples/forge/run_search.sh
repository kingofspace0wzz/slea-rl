set -x

ENGINE=${1:-vllm}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
PROJECT_DIR="$(pwd)"
MODEL_NAME="Qwen2.5-3B-Instruct"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints/forge_search_${MODEL_NAME}"

train_data_size=32
val_data_size=128
group_size=5

# GiGPO config
mode="mean_std_norm" # "mean_norm" or "mean_std_norm"
enable_similarity=True # enable similarity-based GiGPO
similarity_thresh=0.9 # similarity threshold for GiGPO

TRAIN_DATA="$HOME/data/searchR1_processed_direct/train.parquet"
VAL_DATA="$HOME/data/searchR1_processed_direct/test.parquet"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=forge \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/$MODEL_NAME \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.forge.use_step_rewards=True \
    algorithm.forge.step_advantage_w=1.0 \
    algorithm.forge.mode=$mode \
    algorithm.forge.success_threshold=0.5 \
    algorithm.forge.golden_capacity_per_level=100 \
    algorithm.forge.warning_capacity_per_level=50 \
    algorithm.forge.divergence_type=jsd \
    algorithm.forge.distill_temperature=1.0 \
    algorithm.forge.library_save_freq=10 \
    algorithm.forge.checkpoint_dir=${CHECKPOINT_DIR}/library \
    algorithm.gigpo.enable_similarity=$enable_similarity \
    algorithm.gigpo.similarity_thresh=$similarity_thresh \
    env.env_name=search \
    env.seed=0 \
    env.max_steps=4 \
    env.rollout.n=$group_size \
    env.history_length=4 \
    env.search.search_url='http://172.19.0.1:8000/retrieve' \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_search' \
    trainer.experiment_name="forge_${MODEL_NAME}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 \
    trainer.resume_mode=auto \
    trainer.default_local_dir=${CHECKPOINT_DIR} \
    trainer.val_before_train=False $@