results = []

for article_number in range(1, 11):
  f = open(f"results/paraphraser/baseline_no_special/article_{article_number}.md", "r")

  article = f.readlines()

  compression = article[3].replace("Compression: ", "").split('%')[0] + "%"
  rouge2 = article[4].replace("Rouge 2: ", "").strip()
  rougeL = article[5].replace("Rouge L: ", "").strip()
  similarity = article[6].replace("MD Similarity: ", "").strip()
  overall = article[7].replace("Overall: ", "").strip()
  results.append((compression, rouge2, rougeL, similarity, overall))

print("Compression,Rouge 2,Rouge L,MD Similarity,Overall")
for result in results:
  print(f"{result[0]},{result[1]},{result[2]},{result[3]},{result[4]}")