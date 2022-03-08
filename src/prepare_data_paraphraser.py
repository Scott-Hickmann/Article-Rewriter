import pandas as pd
from config_paraphraser import Config
from util import mdEncode, mdRemove
from paraphraser import Paraphraser

config = Config()
paraphraser = Paraphraser(config)

dataframe = pd.read_csv(f'data/{config.rewriter_name}/markdown_data.csv', encoding='utf-8')

dataframe['source_without_markdown'] = [mdRemove(mdEncode(text)[0]) for text in dataframe['source_with_markdown']]

def paraphrase(i: int, text: str):
  paraphrased = paraphraser.paraphrase(paraphraser.encode(text))[0]
  print(f"{i}: {paraphrased}")
  return paraphrased

dataframe['target_without_markdown'] = [paraphrase(i, text) for i, text in enumerate(dataframe['source_without_markdown'])]

dataframe.to_csv(f'data/{config.rewriter_name}/to_annotate_data.csv', encoding='utf-8', index=False)