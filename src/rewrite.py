import pandas as pd
import torch
from config_summarizer import Config as ConfigSummarizer
from config_paraphraser import Config as ConfigParaphraser
from util import mdDecode, mdEncode, mdRemove
import json
from summarizer import Summarizer
from paraphraser import Paraphraser
from evaluate import evaluate

article_input = input('Article number: ')

configSummarizer = ConfigSummarizer()
summarizer = Summarizer(configSummarizer)
summarizer.model.load_state_dict(torch.load(f"models/{configSummarizer.rewriter_name}/main.pt", map_location=configSummarizer.device))
summarizer.model.eval()

configParaphraser = ConfigParaphraser()
paraphraser = Paraphraser(configParaphraser)
paraphraser.model.load_state_dict(torch.load(f"models/{configParaphraser.rewriter_name}/main.pt", map_location=configParaphraser.device))
paraphraser.model.eval()

def rewrite_article(article_number):
  article = json.load(open(f"data/article_{article_number}.json", "r"))
  sources = []
  targets = []
  for i, block in enumerate(article["blocks"]):
    text = block["text"]
    if block["type"] == "paragraph":
      if len(sources) and sources[-1]["type"] == block["type"] and len(f"{sources[-1]['text']}\n{text}") <= summarizer.max_length:
        sources[-1] = {"type": block["type"], "text": f"{sources[-1]['text']}\n\n{text}"}
      else:
        sources.append({"type": block["type"], "text": text})
    else:
      sources.append({"type": block["type"], "text": text})
  for i, source in enumerate(sources):
    print(f'{i}/{len(sources) - 1}')
    if source["type"] == "paragraph":
      mdEncoded, mdLookup = mdEncode(source["text"])
      tmp_result = summarizer.summarize(summarizer.encode(mdEncoded))[0].strip()
      result = paraphraser.paraphrase(paraphraser.encode(tmp_result))[0].strip()
      print(result)
      target = None
      try:
        target = mdDecode(result, mdLookup)
      except:
        print("EXCEPTION")
      if target is not None:
        targets.append(target)
    else:
      targets.append(source["text"])
  original = article["content"]
  print(original)
  print()
  print()
  summarized = "\n\n".join(targets)
  print(summarized)
  print()
  print()
  compression = len(summarized) / len(original) * 100
  compression_str = f"Compression: {compression}% ({len(summarized)}/{len(original)})"
  print(compression_str)

  summary_file = open(f"results/rewriter/article_{article_number}.md", "w")
  summary_file.write(f"""# {article["title"]}

```
{compression_str}
```

{summarized}
""")
  summary_file.close()

if article_input == "":
  for n in range(10):
    rewrite_article(str(n + 1))
else:
  rewrite_article(article_input)