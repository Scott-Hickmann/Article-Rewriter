import pandas as pd
from config import Config
from util import mdEncode, mdRemove
from paraphraser import Paraphraser

config = Config()
paraphraser = Paraphraser(
  device=config.device,
  max_length=config.max_length,
  num_beams=config.num_beams,
  num_beam_groups=config.num_beam_groups,
  diversity_penalty=config.diversity_penalty
)

dataframe = pd.read_csv('data/markdown_data.csv', encoding='utf-8')

dataframe['source_without_markdown'] = [mdRemove(mdEncode(text)[0]) for text in dataframe['source_with_markdown']]

def paraphrase(i: int, text: str):
  paraphrased = paraphraser.paraphrase(paraphraser.encode(text))[0]
  print(f"{i}: {paraphrased}")
  return paraphrased

dataframe['target_without_markdown'] = [paraphrase(i, text) for i, text in enumerate(dataframe['source_without_markdown'])]

dataframe.to_csv('data/fine-tuning_data.csv', encoding='utf-8', index=False)