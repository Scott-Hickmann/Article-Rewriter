import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
  def __init__(self, dataframe, encode):
    self.encode = encode
    self.data = dataframe
    self.source = self.data['source']
    self.target = self.data['target']

  def __len__(self):
    return len(self.target)

  def __getitem__(self, index):
    source = self.encode(str(self.source[index]))
    target = self.encode(str(self.target[index]))

    source_ids = source['input_ids'].squeeze()
    source_mask = source['attention_mask'].squeeze()
    target_ids = target['input_ids'].squeeze()

    return {
      'source_ids': source_ids.to(dtype=torch.long), 
      'source_mask': source_mask.to(dtype=torch.long), 
      'target_ids': target_ids.to(dtype=torch.long),
      'target_ids_y': target_ids.to(dtype=torch.long)
    }