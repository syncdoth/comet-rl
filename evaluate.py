from tqdm import tqdm
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np


@torch.inference_mode()
def evaluate(ppo_trainer, tokenizer, config, eval_dataloader):
    pred_scores = []
    labels = []
    classes = []
    for i, batch in tqdm(enumerate(eval_dataloader)):
        # prepare model inputs
        queries = [x.squeeze() for x in batch['input_ids']]
        responses = [x.squeeze() for x in batch['target_ids']]
        model_inputs = ppo_trainer.prepare_model_inputs(queries, responses)

        if ppo_trainer.is_distributed:
            pad_first = tokenizer.padding_side == "left"

            model_inputs["input_ids"] = ppo_trainer.accelerator.pad_across_processes(
                model_inputs["input_ids"],
                dim=1,
                pad_index=tokenizer.pad_token_id,
                pad_first=pad_first)
            model_inputs["attention_mask"] = ppo_trainer.accelerator.pad_across_processes(
                model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first)
            if ppo_trainer.is_encoder_decoder:
                model_inputs["decoder_input_ids"] = ppo_trainer.accelerator.pad_across_processes(
                    model_inputs["decoder_input_ids"],
                    dim=1,
                    pad_index=tokenizer.pad_token_id,
                    pad_first=pad_first,
                )
                model_inputs[
                    "decoder_attention_mask"] = ppo_trainer.accelerator.pad_across_processes(
                        model_inputs["decoder_attention_mask"],
                        dim=1,
                        pad_index=0,
                        pad_first=pad_first)
        # do forward to get logits
        ppo_trainer.model.eval()
        lm_logits = ppo_trainer.model(**model_inputs)[0]

        # setup target sequence
        if ppo_trainer.is_encoder_decoder:
            lm_labels = torch.where(model_inputs['decoder_input_ids'] == tokenizer.pad_token_id,
                                    -100, model_inputs['decoder_input_ids'])
            target_len = model_inputs['decoder_attention_mask'][:, 1:].sum(dim=-1)
        else:
            lm_labels = torch.where(model_inputs['input_ids'] == tokenizer.pad_token_id, -100,
                                    model_inputs['input_ids'])
            target_len = model_inputs['attention_mask'][:, 1:].sum(dim=-1)

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
        shift_labels = lm_labels[..., 1:].contiguous().view(-1)
        # Flatten the tokens
        # compute loss (perplexity) as pred score
        loss = F.cross_entropy(shift_logits, shift_labels,
                               reduction='none').view(model_inputs['input_ids'].size(0), -1)
        loss = torch.div(loss.sum(dim=-1), target_len)  # [B,]
        pred = -loss
        pred_scores.extend(pred.tolist())

        label = torch.cat(batch["label"], dim=0).tolist()
        clss = torch.cat(batch["clss"], dim=0).tolist()
        labels.extend(label)
        classes.extend(clss)

    classes = np.array(classes)
    predicted_scores = np.array(pred_scores)
    labels = np.array(labels)

    clss_scores = {}
    clss_num = {}
    for clss in ["test_set", "cs_head", "all_head", "adv"]:
        idx = classes == eval_dataloader.dataset.clss_map[clss]
        clss_num[clss] = sum(idx)
        clss_scores[clss] = roc_auc_score(labels[idx], predicted_scores[idx])

    clss_num["overall"] = len(labels)
    clss_scores["overall"] = roc_auc_score(labels, predicted_scores)

    return clss_scores, clss_num
