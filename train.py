import os

from tqdm import tqdm
import torch

from evaluate import evaluate


def train(ppo_trainer,
          config,
          tokenizer,
          rm_tokenizer=None,
          reward_model=None,
          generation_kwargs=None,
          output_length_sampler=None,
          eval_dataloader=None):
    global_step = 0

    os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
    for epoch in range(config.num_epochs):
        for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
            query_tensors = batch["input_ids"]
            # use RM
            if rm_tokenizer is not None and reward_model is not None:
                responses, rewards = compute_reward(ppo_trainer, tokenizer, rm_tokenizer,
                                                    reward_model, generation_kwargs,
                                                    output_length_sampler, query_tensors)
                query_tensors = [q.squeeze(0) for q in query_tensors]
                batch["response"] = responses
            # Just use label as reward
            else:
                batch["response"] = [tokenizer.decode(r.squeeze()) for r in batch["target_ids"]]
                batch["query"] = [tokenizer.decode(q.squeeze()) for q in query_tensors]
                query_tensors = [q.squeeze(0) for q in query_tensors]
                response_tensors = [r.squeeze(0) for r in batch["target_ids"]]
                rewards = [l.squeeze(0).float() for l in batch["label"]]

            # Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

            global_step += 1
            if config.eval_step > 0 and global_step % config.eval_step == 0:
                assert eval_dataloader is not None
                scores, support = evaluate(ppo_trainer, tokenizer, config, eval_dataloader)
                if ppo_trainer.accelerator.is_main_process:
                    ppo_trainer.accelerator.log({'validation': scores})

            if global_step % config.save_step == 0:
                if ppo_trainer.accelerator.is_main_process:
                    ppo_trainer.save_pretrained(config.model_save_path)

            if global_step == config.max_step:
                print(f"stop training at {config.max_step} steps.")
                return


def compute_reward(ppo_trainer,
                   tokenizer,
                   rm_tokenizer,
                   reward_model,
                   generation_kwargs,
                   output_length_sampler,
                   query_tensors,
                   attribute_idx=0):
    response_tensors = []
    for query in query_tensors:
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        response = ppo_trainer.generate(query, **generation_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])
    responses = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    # Compute sentiment score # noqa
    rm_inputs = rm_tokenizer(responses, padding=True, truncation=True,
                             return_tensors="pt").to(ppo_trainer.accelerator.device)
    logits = reward_model(**rm_inputs).logits.float()
    labels = (logits[:, attribute_idx]).tolist()

    rewards = [torch.tensor(output) for output in labels]

    return responses, rewards