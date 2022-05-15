import pandas as pd
import torch
from sklearn.utils import shuffle
from typing import List, Dict, Union
from torch.utils.data import Dataset
from tqdm.auto import tqdm


def read_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t')
    df = df.fillna('')

    df_train_toxic = []
    df_train_neutral = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc='Load dataset from file'):
        references = row[['neutral_comment1', 'neutral_comment2', 'neutral_comment3']].tolist()

        for reference in references:
            if len(reference) > 0:
                df_train_toxic.append(row['toxic_comment'])
                df_train_neutral.append(reference)
            else:
                break

    df = pd.DataFrame({
        'toxic_comment': df_train_toxic,
        'neutral_comment': df_train_neutral
    })

    df = shuffle(df)
    return df


class PairsDataset(Dataset):
    def __init__(self, x, y, neutral_toxicity=None):
        self.x = x
        self.y = y
        self._neutral_toxicity = neutral_toxicity

    def __getitem__(self, idx):
        assert idx < len(self.x['input_ids'])
        item = {key: val[idx] for key, val in self.x.items()}
        item['decoder_attention_mask'] = self.y['attention_mask'][idx]
        item['labels'] = self.y['input_ids'][idx]
        if self._neutral_toxicity is not None:
            item['neutral_toxicity'] = self._neutral_toxicity[idx]
        return item

    @property
    def n(self):
        return len(self.x['input_ids'])

    def __len__(self):
        return self.n  # * 2


class DataCollatorWithPadding:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            features,
            padding=True,
        )
        ybatch = self.tokenizer.pad(
            {'input_ids': batch['labels'], 'attention_mask': batch['decoder_attention_mask']},
            padding=True,
        )
        batch['labels'] = ybatch['input_ids']
        batch['decoder_attention_mask'] = ybatch['attention_mask']

        return {k: torch.tensor(v) for k, v in batch.items()}
