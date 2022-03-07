import pandas as pd

dataframe = pd.read_csv('data/predictions.csv', encoding='utf-8')

for text in dataframe['generated']:
  print(text)