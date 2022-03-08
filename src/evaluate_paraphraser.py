import pandas as pd
import torch
from config_paraphraser import Config
from util import mdDecode, mdEncode, mdRemove
import json
import nltk
from paraphraser import Paraphraser
from evaluate import evaluate

nltk.download('punkt')

article_input = input('Article number: ')

config = Config()
paraphraser = Paraphraser(config)
# paraphraser.model.load_state_dict(torch.load(f"models/{config.rewriter_name}/main.pt"))
# paraphraser.model.eval()
naive_paraphraser = Paraphraser(config)

def paraphrase_article(article_number):
  article = json.load(open(f"data/article_{article_number}.json", "r"))
  sources = []
  targets = []
  generated = []
  expected = []
  original = ""
  paraphrased = ""
  for i, block in enumerate(article["blocks"]):
    print(f'{i}/{len(article["blocks"]) - 1}')
    if block["type"] == "paragraph":
      for sentence in nltk.tokenize.sent_tokenize(block["text"]):
        original += sentence
        mdEncoded, mdLookup = mdEncode(sentence)
        encoded = paraphraser.encode(mdEncoded)
        result = paraphraser.paraphrase(encoded)[0].strip()
        print(result)
        target = None
        try:
          target = mdDecode(result, mdLookup)
        except:
          print("EXCEPTION")
        if target is not None:
          source_without_markdown = mdRemove(mdEncoded)
          expected_res = naive_paraphraser.paraphrase(naive_paraphraser.encode(source_without_markdown))[0].strip()
          targets.append(target)
          paraphrased += target
          generated.append(mdRemove(result))
          expected.append(expected_res)
  original = "".join([source["text"] for source in sources])
  print(original)
  print()
  print()
  print(paraphrased)
  print()
  print()
  compression_str = f"Compression: {len(paraphrased) / len(original)} ({len(paraphrased)}/{len(original)})"
  print(compression_str)
  final_df = pd.DataFrame({'generated': generated, 'expected': expected})
  rouge2_score, rougeL_score = evaluate(final_df)

  paraphrase_file = open(f"results/{config.rewriter_name}/article_{article_number}.md", "w")
  paraphrase_file.write(f"""# {article["title"]}

```
{compression_str}
Rouge 2: {rouge2_score}%
Rouge L: {rougeL_score}%
```

{result}
""")
  paraphrase_file.close()

if article_input == "":
  for n in range(10):
    paraphrase_article(str(n + 1))
else:
  paraphrase_article(article_input)