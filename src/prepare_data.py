import json
import pandas as pd
from config import Config
from util import mdEncode, mdRemove
from summarizer import Summarizer

config = Config()
summarizer = Summarizer(
  device=config.device,
  min_length=config.min_length,
  max_length=config.max_length,
  summary_max_length=config.summary_max_length,
  num_beams=config.num_beams,
  length_penalty=config.length_penalty
)

dataframe = pd.read_csv('data/raw_data.csv', encoding='utf-8')

def summarize(i: int, text: str):
  summarized = summarizer.summarize(summarizer.encode(text))[0]
  print(f"{i}: {summarized}")
  return summarized

sources_with_markdown = []
for raw_article in dataframe['raw_articles']:
  article = json.loads(raw_article)
  sources = []
  for i, block in enumerate(article["blocks"]):
    if block["type"] == "paragraph":
      text = block["text"]
      if len(sources) and sources[-1]["lastIndex"] == i - 1 and len(f"{sources[-1]['text']}\n{text}") <= summarizer.max_length:
        sources[-1] = {"lastIndex": i, "text": f"{sources[-1]['text']}\n\n{text}"}
      else:
        sources.append({"lastIndex": i, "text": text})
  for source in sources:
    sources_with_markdown.append(source["text"])

dataframe = pd.DataFrame()
dataframe['source_with_markdown'] = sources_with_markdown
dataframe['source_without_markdown'] = [mdRemove(mdEncode(text)[0]) for text in sources_with_markdown]
dataframe['target_without_markdown'] = [summarize(i, text) for i, text in enumerate(dataframe['source_without_markdown'])]

dataframe.to_csv('data/to_annotate_data.csv', encoding='utf-8', index=False)