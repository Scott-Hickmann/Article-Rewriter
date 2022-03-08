import pandas as pd
import torch
from config import Config
from util import mdDecode, mdEncode, mdRemove
from summarizer import Summarizer
import json
from evaluate import evaluate

article_input = input('Article number: ')

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
naive_summarizer = Summarizer(
  device=config.device,
  min_length=config.min_length,
  max_length=config.max_length,
  summary_max_length=config.summary_max_length,
  num_beams=config.num_beams,
  length_penalty=config.length_penalty
)

def summarize_article(article_number):
  article = json.load(open(f"data/article_{article_number}.json", "r"))
  sources = []
  targets = []
  generated = []
  expected = []
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
    target = None
    try:
      target = mdDecode(result, mdLookup)
    except:
      print("EXCEPTION")
    if target is not None:
      source_without_markdown = mdRemove(mdEncoded)
      expected_res = naive_summarizer.summarize(naive_summarizer.encode(source_without_markdown))[0].strip()
      targets.append(target)
      generated.append(mdRemove(result))
      expected.append(expected_res)
  original = "".join([source["text"] for source in sources])
  print(original)
  print()
  print()
  result = "\n\n".join(targets)
  print(result)
  print()
  print()
  compression_str = f"Compression: {len(result) / len(original)} ({len(result)}/{len(original)})"
  print(compression_str)
  final_df = pd.DataFrame({'generated': generated, 'expected': expected})
  rouge2_score, rougeL_score = evaluate(final_df)

  summary_file = open(f"data/article_{article_number}_summary.md", "w")
  summary_file.write(f"""# {article["title"]}

```
{compression_str}
Rouge 2: {rouge2_score}%
Rouge L: {rougeL_score}%
```

{result}
""")
  summary_file.close()

if article_input == "":
  for n in range(10):
    summarize_article(str(n + 1))
else:
  summarize_article(article_input)