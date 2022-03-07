import torch
from config import Config
from util import mdDecode, mdEncode
from summarizer import Summarizer
import json

article = json.load(open(f"data/article_{input('Article number: ')}.json", "r"))

config = Config()
summarizer = Summarizer(
  device=config.device,
  min_length=config.min_length,
  max_length=config.max_length,
  summary_max_length=config.summary_max_length,
  num_beams=config.num_beams,
  length_penalty=config.length_penalty
)
summarizer.model.load_state_dict(torch.load("models/main.pt"))
summarizer.model.eval()

sources = []
targets = []
for i, block in enumerate(article["blocks"]):
  if block["type"] == "paragraph":
    text = block["text"]
    if len(sources) and sources[-1]["lastIndex"] == i - 1 and len(f"{sources[-1]['text']}\n{text}") <= summarizer.max_length:
      sources[-1] = {"lastIndex": i, "text": f"{sources[-1]['text']}\n\n{text}"}
    else:
      sources.append({"lastIndex": i, "text": text})
for i, source in enumerate(sources):
  print(f'{i}/{len(sources) - 1}')
  mdEncoded, mdLookup = mdEncode(source["text"])
  encoded = summarizer.encode(mdEncoded)
  result = summarizer.summarize(encoded)[0].strip()
  print(result)
  try:
    targets.append(mdDecode(result, mdLookup))
  except:
    print("EXCEPTION")
print("".join([source["text"] for source in sources]))
print()
print()
result = "\n\n".join(targets)
print(result)
print()
print()
print(len(result))