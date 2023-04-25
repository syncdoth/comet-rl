from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import torch

from torch.optim import Adam
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    AutoModelForSequenceClassification,
)
from trl import (
    AutoModelForCausalLMWithValueHead,
    AutoModelForSeq2SeqLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    create_reference_model,
    set_seed,
)
from trl.core import LengthSampler

from data import CKBPPPODataset, collator, get_loader, ASER_RELATIONS_2NL, CS_RELATIONS_2NL
from evaluate import evaluate
from train import train

torch.backends.cuda.matmul.allow_tf32 = True


@dataclass
class ScriptArguments(PPOConfig):
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="comet_pretrained/gpt2xl-comet-atomic-2020",
                                      metadata={"help": "the model name"})
    # logging (wandb)
    log_with: Optional[str] = field(default="wandb",
                                    metadata={"help": "use 'wandb' to log with wandb"})
    tracker_project_name: str = "comet-ppo"
    exp_name: str = "default param"
    # training
    num_epochs: int = 1
    learning_rate: Optional[float] = field(default=2e-5, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=4, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"})
    # steps
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"})
    save_step: int = 100
    max_step: int = field(default=-1, metadata={"help": "-1 means Null."})
    eval_step: int = field(default=-1, metadata={"help": "-1 means no eval"})
    # paths
    model_save_path: Optional[str] = field(
        default="checkpoints/gpt2xl-comet-atomic-2020-ppo",
        metadata={"help": "the path to save the model"},
    )
    train_data_path: str = 'ckbp_data/train.csv'
    evaluation_data_path: str = 'ckbp_data/ckbp2.0.csv'
    # reward model
    use_rm: bool = False
    reward_model_id: str = ""
    # relations
    add_rel_token: bool = True
    use_nl_rel: bool = False


def main():
    parser = HfArgumentParser(ScriptArguments)
    config = parser.parse_args_into_dataclasses()[0]

    config.tracker_kwargs = {'wandb': {'name': config.exp_name}}

    # set seed before initializing value head for deterministic eval
    set_seed(config.seed)

    ########################### Load Tokenizer #################################
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    if config.add_rel_token:
        tokenizer.add_special_tokens({
            'additional_special_tokens':
                list(CS_RELATIONS_2NL.keys()) + list(ASER_RELATIONS_2NL.keys())
        })
    ############################################################################

    ########################### Load Data ######################################
    # train dataset
    train_file = pd.read_csv(config.train_data_path)
    train_file = train_file.dropna(subset=['head', 'tail', 'relation'])
    train_dataset = CKBPPPODataset(train_file, tokenizer)

    # eval_dataset
    eval_params = {'batch_size': config.batch_size, 'shuffle': False}
    infer_file = pd.read_csv(config.evaluation_data_path)
    dev_loader = get_loader(infer_file[infer_file['split'] == 'dev'], tokenizer, **eval_params)
    tst_loader = get_loader(infer_file[infer_file['split'] == 'tst'], tokenizer, **eval_params)
    ############################################################################

    ############################ Load Model ####################################
    if 'bart' in config.model_name.lower() or 't5' in config.model_name.lower():
        model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16)
        model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16)
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

    if config.add_rel_token:
        model.pretrained_model.resize_token_embeddings(len(tokenizer))

    ref_model = create_reference_model(model, num_shared_layers=None)
    ############################################################################

    # We make sure to use `Adam` optimizer on the model parameters that require gradients.
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_collator=collator,
        optimizer=optimizer,
    )

    if config.use_rm:
        rm_tokenizer = AutoTokenizer.from_pretrained(config.reward_model_id)
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            config.reward_model_id, torch_dtype=torch.bfloat16).to(ppo_trainer.accelerator.device)

        # We then define the arguments to pass to the `generate` function. These arguments
        # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
        # the `generate` function of the trained model.
        generation_kwargs = {
            "min_length": -1,
            "top_p": 0.95,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
        }
        output_min_length = 20
        output_max_length = 30
        output_length_sampler = LengthSampler(output_min_length, output_max_length)

        train(ppo_trainer,
              config,
              tokenizer=tokenizer,
              rm_tokenizer=rm_tokenizer,
              reward_model=reward_model,
              generation_kwargs=generation_kwargs,
              output_length_sampler=output_length_sampler,
              eval_dataloader=dev_loader)
    else:
        train(ppo_trainer, config, tokenizer=tokenizer, eval_dataloader=dev_loader)

    scores, support = evaluate(ppo_trainer, tokenizer, config, tst_loader)
    if ppo_trainer.accelerator.is_main_process:
        ppo_trainer.accelerator.log({'test': scores})


if __name__ == "__main__":
    main()
