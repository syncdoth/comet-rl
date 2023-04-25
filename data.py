import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Commonsense relationships
CS_RELATIONS_2NL = {
    "oEffect": "then, PersonY will",
    "xEffect": "then, PersonX will",
    "general Effect": "then, other people or things will",
    "oWant": "then, PersonY wants to ",
    "xWant": "then, PersonX wants to",
    "general Want": "then, other people or things want to",
    "oReact": "then, PersonY feels",
    "xReact": "then, PersonX feels",
    "general React": "then, other people or things feel",
    "xAttr": "PersonX is seen as",
    "xNeed": "but before, PersonX needed",
    "xIntent": "because PersonX wanted",
    "isBefore": "happens before",
    "isAfter": "happens after",
    "HinderedBy": "can be hindered by",
    "xReason": "because",
    "Causes": "causes ",
    "HasSubEvent": "includes the event or action",
}

CS_RELATIONS = {
    "all": [
        'xAttr', 'xReact', 'xWant', 'xEffect', 'xNeed', 'oWant', 'oReact', 'xIntent', 'oEffect',
        'HinderedBy', 'Causes', 'isBefore', 'isAfter', 'general Effect', 'HasSubEvent',
        'general Want', 'general React', 'xReason'
    ],
    "atomic": [
        'xAttr', 'xReact', 'xWant', 'xEffect', 'xNeed', 'oWant', 'oReact', 'xIntent', 'oEffect',
        'HinderedBy', 'isBefore', 'isAfter'
    ],
    "glucose": [
        'xAttr', 'xReact', 'xWant', 'xEffect', 'oWant', 'oReact', 'xIntent', 'oEffect', 'Causes',
        'general Effect', 'general Want', 'general React'
    ],
    "cn": ['Causes', 'xReason', 'HasSubEvent'],
}

ASER_RELATIONS_2NL = {
    'Co_Occurrence': 'appears together with',
    'Conjunction': 'and',
    'Contrast': 'on the contrary,',
    'Synchronous': 'happens at the same time as',
    'Condition': 'as long as',
    'Reason': 'happens because of',
    'Result': 'therefore,',
    'Precedence': 'happens before',
    'Succession': 'happens after',
    'Alternative': 'is alternative to',
    'Concession': 'although',
    'Restatement': 'in other words,',
    'ChosenAlternative': 'instead of',
    'Exception': 'except for',
    'Instantiation': 'for instance,',
}


def get_loader(dataframe, tokenizer, **params):
    dataset = CKBPPPODataset(dataframe, tokenizer)
    return DataLoader(dataset, collate_fn=collator, **params)


class CKBPPPODataset(Dataset):
    """A dataset class for multi task setting: loads all relations.
    """

    def __init__(self, dataframe, tokenizer):
        """
        Args:
            dataframe (pd.DataFrame):
                pandas dataframe object for edges
            graph_or_path (str or nx.Graph):
                path of the input networkx graph file / or the graph file
                Needed if rel2id, id2rel, node2id, id2node are None.
        """

        # Load edges
        self.data = dataframe.reset_index()
        # filter too short tails < 2
        # NOTE: naive tokenization with whitespace
        self.data['tail_length'] = self.data['tail'].apply(lambda x: len(x.split(' ')))
        self.data = self.data[self.data['tail_length'] >= 2]
        self.data = self.data.reset_index()

        self.edges = self.data[["head", "tail", "relation"]].values
        self.tokenizer = tokenizer

        # Load label and class
        if "label" in self.data:
            self.labels = self.data["label"]
        elif "majority_vote" in self.data:  # for evaluation_set.csv
            self.labels = self.data["majority_vote"]
        else:
            # useful when scoring pseudo edges.
            self.labels = pd.Series([0 for i in range(len(self.data))])

        # for evaluation set only
        if "class" in self.data:
            self.clss = self.data["class"]
        else:
            print("No class labels in the dataset")
            self.clss = pd.Series(["" for _ in range(len(self.data))])
        self.clss_map = {'': -1, 'test_set': 0, 'cs_head': 1, 'all_head': 2, 'adv': 3}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        edge = self.edges[idx].tolist()
        input_ids = self.tokenizer(f"{edge[0]} {edge[2]} [GEN]", return_tensors='pt').input_ids
        target_ids = self.tokenizer(f"{edge[1]} [EOS]", return_tensors='pt').input_ids

        item = {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'label': torch.LongTensor([self.labels[idx]]),
            'clss': torch.LongTensor([self.clss_map[self.clss[idx]]])
        }

        return item


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])
