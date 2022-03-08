import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from util import mdDecode, mdEncode
from dataset import CustomDataset
from evaluate import evaluate

def finetune(config, rewriter):
  dataframe = pd.read_csv(f'data/{config.rewriter_name}/annotated_data.csv', encoding='utf-8')
  train, val = train_test_split(dataframe, test_size=0.2, random_state=config.seed)

  def get_dataset(data):
    md_source_with_markdown = [mdEncode(text) for text in data['source_with_markdown']]
    return CustomDataset({
      "source": [md_data[0] for md_data in md_source_with_markdown],
      "target": [mdEncode(text, md_source_with_markdown[i][1])[0] for i, text in enumerate(data['target_with_markdown'])],
    }, rewriter.encode), md_source_with_markdown

  training_set, _ = get_dataset(train)
  val_set, md_val_source_with_markdown = get_dataset(val)

  print(training_set.target, val_set.target)

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

  optimizer = torch.optim.Adam(rewriter.model.parameters(), lr=config.learning_rate)

  def decode_batch(md_generated, md_expected):
    generated = []
    expected = []
    for a_md_generated, a_md_expected in zip(md_generated, md_expected):
      expected_index = val_set.target.index(a_md_expected)
      md_lookup = md_val_source_with_markdown[expected_index][1]
      try:
        generated.append(mdDecode(a_md_generated, md_lookup))
        expected.append(mdDecode(a_md_expected, md_lookup))
      except:
        generated.append(a_md_generated)
        expected.append(a_md_expected)
    return generated, expected

  print('Fine-tuning the model on our dataset')
  best_score = 0
  for epoch in range(config.train_epochs):
    rewriter.train(epoch, training_loader, optimizer)
    print()
    print('Evaluating')
    md_generated, md_expected = rewriter.validate(val_loader)
    generated, expected = decode_batch(md_generated, md_expected)
    final_df = pd.DataFrame({'generated': generated, 'expected': expected})
    final_df.to_csv(f'data/{config.rewriter_name}/predictions.csv', encoding='utf-8', index=False)
    score, _ = evaluate(final_df)
    if score > best_score:
      print("New best score, saving to dataset")
      best_score = score
      torch.save(rewriter.model.state_dict(), f'models/{config.rewriter_name}/main.pt')
    print()