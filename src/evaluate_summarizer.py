import pandas as pd
import torch
from config_summarizer import Config
from util import mdDecode, mdEncode, mdRemove
import json
from summarizer import Summarizer
from evaluate import evaluate

article_input = input('Article number: ')

config = Config()
summarizer = Summarizer(config)
summarizer.model.load_state_dict(torch.load(f"models/{config.rewriter_name}/main.pt"))
summarizer.model.eval()
naive_summarizer = Summarizer(config)

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
  summarized = "\n\n".join(targets)
  print(result)
  print()
  print()
  compression = len(summarized) / len(original) * 100
  compression_str = f"Compression: {compression}% ({len(summarized)}/{len(original)})"
  print(compression_str)
  final_df = pd.DataFrame({'generated': generated, 'expected': expected})
  rouge2_score, rougeL_score = evaluate(final_df)

  summary_file = open(f"results/{config.rewriter_name}/article_{article_number}.md", "w")
  summary_file.write(f"""# {article["title"]}

```
{compression_str}
Rouge 2: {rouge2_score}%
Rouge L: {rougeL_score}%
```

{result}
""")
  summary_file.close()
  return compression, rouge2_score, rougeL_score

results = []
if article_input == "":
  for n in range(10):
    results.append(summarize_article(str(n + 1)))
else:
  results.append(summarize_article(article_input))

print("Compression,Rouge 2,Rouge L")
for result in results:
  print(f"{result[0]}%,{result[1]}%,{result[2]}%")