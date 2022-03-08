import pandas as pd
from datasets import load_metric
from config import Config

config = Config()

metric = load_metric("rouge")

dataframe = pd.read_csv('data/predictions.csv', encoding='utf-8')

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