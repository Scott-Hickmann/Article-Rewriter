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
paraphraser.model.load_state_dict(torch.load(f"models/{config.rewriter_name}/main.pt"))
paraphraser.model.eval()
naive_paraphraser = Paraphraser(config)

def paraphrase_article(article_number):
  article = json.load(open(f"data/article_{article_number}.json", "r"))
  targets = []
  generated = []
  expected = []
  original = ""
  paraphrased = ""
  md_generated = []
  md_original = []
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
          paraphrased += target + " "
          generated.append(mdRemove(result))
          expected.append(expected_res)
          md_generated.append(result)
          md_original.append(mdEncoded)
  print(original)
  print()
  print()
  print(paraphrased)
  print()
  print()
  compression = len(paraphrased) / len(original) * 100
  compression_str = f"Compression: {compression}% ({len(paraphrased)}/{len(original)})"
  print(compression_str)
  final_df = pd.DataFrame({'generated': generated, 'expected': expected, 'generated_md': md_generated, 'original_md': md_original})
  final_df.to_csv(f'results/{config.rewriter_name}/article_{article_number}.csv', encoding='utf-8', index=False)
  overall_score, rouge2_score, rougeL_score, md_similarity_score = evaluate(final_df)

  paraphrase_file = open(f"results/{config.rewriter_name}/article_{article_number}.md", "w")
  paraphrase_file.write(f"""# {article["title"]}

```
{compression_str}
Rouge 2: {rouge2_score}%
Rouge L: {rougeL_score}%
MD Similarity: {md_similarity_score}%
Overall: {overall_score}%
```

{paraphrased}
""")
  paraphrase_file.close()
  return compression, rouge2_score, rougeL_score, md_similarity_score, overall_score

results = []
if article_input == "":
  for n in range(10):
    results.append(paraphrase_article(str(n + 1)))
else:
  results.append(paraphrase_article(article_input))

print("Compression,Rouge 2,Rouge L,MD Similarity,Overall")
for result in results:
  print(f"{result[0]}%,{result[1]}%,{result[2]}%,{result[3]}%,{result[4]}%")