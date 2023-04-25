import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class KGBERTClassifier(nn.Module):

    def __init__(self, model_name, dropout=0, **model_load_kwargs):
        super().__init__()

        self.model = AutoModel.from_pretrained(model_name, **model_load_kwargs)
        self.model_type = self.model.config.model_type

        try:
            self.emb_size = self.model.config.d_model  # bart
        except:
            self.emb_size = self.model.config.hidden_size  # roberta/bert

        torch_dtype = model_load_kwargs.get("torch_dtype", None)
        self.linear = nn.Linear(self.emb_size, 2, dtype=torch_dtype)

        # self.dropout = nn.Dropout(p=dropout)

    def get_lm_embedding(self, tokens):
        """
            Input_ids: tensor (num_node, max_length)

            output:
                tensor: (num_node, emb_size)
        """
        outputs = self.model(tokens['input_ids'], attention_mask=tokens['attention_mask'])

        if self.model_type == "bart":
            # embedding of [EOS] in the decoder
            eos_mask = tokens['input_ids'].eq(self.model.config.eos_token_id)

            if torch.any(eos_mask.sum(1) > 1):
                raise ValueError("All examples must have only one <eos> tokens.")
            sentence_representation = outputs[0][eos_mask, :].view(outputs[0].size(0), -1,
                                                                   outputs[0].size(-1))[:, -1, :]
        else:
            # embedding of the [CLS] tokens
            sentence_representation = outputs[0][:, 0, :]

        return sentence_representation

    def forward(self, tokens):
        """
            tokens:
        """

        embs = self.get_lm_embedding(tokens)  # (batch_size, emb_size)

        logits = self.linear(embs)  # (batch_size, 2)

        # return self.dropout(logits)
        return logits


SUPPORTED_RM_TYPES = {'kgbert': KGBERTClassifier}


def load_reward_model(model_type,
                      ptlm_name,
                      saved_model_path=None,
                      special_token_list=None,
                      device='cpu',
                      **model_load_kwargs):
    if model_type not in SUPPORTED_RM_TYPES:
        raise ValueError(
            f'model type {model_type} is not supported. Should be one of {list(SUPPORTED_RM_TYPES.keys())}.'
        )

    model = SUPPORTED_RM_TYPES[model_type](ptlm_name, **model_load_kwargs).to(device)
    tokenizer = AutoTokenizer.from_pretrained(ptlm_name)
    if special_token_list is not None:
        tokenizer.add_special_tokens({
            'additional_special_tokens': special_token_list,
        })
        model.model.resize_token_embeddings(len(tokenizer))

    if saved_model_path is not None:
        model.load_state_dict(torch.load(saved_model_path))

    return model, tokenizer
