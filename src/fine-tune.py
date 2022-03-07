import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from config import Config
from util import mdEncode
from summarizer import Summarizer
from dataset import CustomDataset

config = Config()
summarizer = Summarizer(
  device=config.device,
  min_length=config.min_length,
  max_length=config.max_length,
  summary_max_length=config.summary_max_length,
  num_beams=config.num_beams,
  length_penalty=config.length_penalty
)

dataframe = pd.read_csv('data/annotated_data.csv', encoding='utf-8')
train, val = train_test_split(dataframe, test_size=0.2)

def get_dataset(data):
  md_source_with_markdown = [mdEncode(text) for text in data['source_with_markdown']]
  return CustomDataset({
    "source": [md_data[0] for md_data in md_source_with_markdown],
    "target": [mdEncode(text, md_source_with_markdown[i][1])[0] for i, text in enumerate(data['target_with_markdown'])],
  }, summarizer.encode)

training_set = get_dataset(train)
val_set = get_dataset(val)

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

optimizer = torch.optim.Adam(summarizer.model.parameters(), lr=config.learning_rate)

print('Fine-tuning the model on our dataset')
for epoch in range(config.train_epochs):
  summarizer.train(epoch, training_loader, optimizer)
print()

print('Generating paraphrases on our fine-tuned model for the validation dataset and saving it in a dataframe')
generated, actual = summarizer.validate(val_loader)
final_df = pd.DataFrame({'generated': generated, 'expected': actual})
final_df.to_csv('data/predictions.csv')
print('Output Files generated for review')