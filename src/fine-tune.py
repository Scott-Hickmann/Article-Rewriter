import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from config import Config
from util import mdEncode
from paraphraser import Paraphraser
from dataset import CustomDataset

config = Config()
paraphraser = Paraphraser(
  device=config.device,
  max_length=config.max_length,
  num_beams=config.num_beams,
  num_beam_groups=config.num_beam_groups,
  diversity_penalty=config.diversity_penalty
)

dataframe = pd.read_csv('data/annotated_data.csv', encoding='utf-8')
train, val = train_test_split(dataframe, test_size=0.2)

training_set = CustomDataset({
  "source": [mdEncode(text)[0] for text in train['source_with_markdown']],
  "target": [mdEncode(text)[0] for text in train['target_with_markdown']],
}, paraphraser.encode)

val_set = CustomDataset({
  "source": [mdEncode(text)[0] for text in val['source_with_markdown']],
  "target": [mdEncode(text)[0] for text in val['target_with_markdown']],
}, paraphraser.encode)

train_params = {
  'batch_size': config.train_batch_size,
  'shuffle': True,
  'num_workers': 0
}

val_params = {
  'batch_size': config.val_batch_size,
  'shuffle': False,
  'num_workers': 0
}

training_loader = DataLoader(training_set, **train_params)
val_loader = DataLoader(val_set, **val_params)

optimizer = torch.optim.Adam(paraphraser.model.parameters(), lr=config.learning_rate)

print('Fine-tuning the model on our dataset')
for epoch in range(config.train_epochs):
  paraphraser.train(epoch, training_loader, optimizer)
print()

print('Generating paraphrases on our fine-tuned model for the validation dataset and saving it in a dataframe')
generated, actual = paraphraser.validate(val_loader)
final_df = pd.DataFrame({'generated': generated, 'actual': actual})
final_df.to_csv('data/predictions.csv')
print('Output Files generated for review')