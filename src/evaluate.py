import pandas as pd
from datasets import load_metric
from config import Config

def evaluate(dataframe):
  metric = load_metric("rouge")

  for index, row in dataframe.iterrows():
    generated = row['generated']
    expected = row['expected']
    if pd.isna(generated):
      generated = ""
    metric.add(prediction=generated, reference=expected)

  final_score = metric.compute(rouge_types=["rouge2", "rougeL"])
  rouge2 = final_score["rouge2"].mid
  rougeL = final_score["rougeL"].mid
  print(f"Rouge 2: {rouge2.fmeasure * 100}%")
  print(f"Rouge L: {rougeL.fmeasure * 100}%")
  return rouge2.fmeasure

if __name__ == "__main__":
  config = Config() # To initialize random seed
  dataframe = pd.read_csv('data/predictions.csv', encoding='utf-8')
  evaluate(dataframe)