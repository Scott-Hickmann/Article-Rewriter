import re
import pandas as pd
from datasets import load_metric
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader

word2vec_model = gensim.downloader.load("glove-wiki-gigaword-100")

def avg_words_vector(words, model, num_features):
  featureVec = np.zeros((num_features, 1), dtype="float32")
  nwords = 0
  for word in words:
    if word in model.index_to_key:
      nwords += 1
      featureVec = np.add(featureVec, model[word])
  if nwords > 0:
    featureVec = np.divide(featureVec, nwords)
  return featureVec

def get_words_similarity(sentence_1, sentence_2):
  sentence_1_avg_vector = avg_words_vector(sentence_1.split(), model=word2vec_model, num_features=100)
  sentence_2_avg_vector = avg_words_vector(sentence_2.split(), model=word2vec_model, num_features=100)
  return 2 * (max(0.5, min(cosine_similarity(sentence_1_avg_vector, sentence_2_avg_vector)[0][0], 1.0)) - 0.5)

def evaluate(dataframe):
  metric = load_metric("rouge")

  for _, row in dataframe.iterrows():
    generated = row['generated']
    expected = row['expected']
    if pd.isna(generated):
      generated = ""
    metric.add(prediction=generated, reference=expected)

  final_score = metric.compute(rouge_types=["rouge2", "rougeL"])
  rouge2 = final_score["rouge2"].mid
  rougeL = final_score["rougeL"].mid

  total_similarity_score = 0
  total_max_similarity_score = 0

  for _, row in dataframe.iterrows():
    text = row['generated_md']
    original = row['original_md']
    for n in range(100):
      tag = f"\(MD{n}\)"
      tagsFound = [(m.start(0), m.end(0), m.group()) for m in re.finditer(tag, text)]

      if len(tagsFound) > 0 and n > 0:
        tagSearcher = re.compile(f'{tag}(.*?){tag}')
        tags = tagSearcher.findall(text)
        original_tags = tagSearcher.findall(original)
        totat_similarity = 0
        for tag in tags:
          totat_similarity += max([get_words_similarity(tag, original_tag) for original_tag in original_tags])
        avg_similarity = totat_similarity / len(tags)
        total_similarity_score += avg_similarity
        total_max_similarity_score += 1

      if n == 0:
        total_max_similarity_score += 1
        if len(tagsFound) >= 2 and tagsFound[0][0] == 0 and tagsFound[-1][1] == len(text): # Severly penalized missing (MD0) enclosure
          total_similarity_score += 1
  
  if total_max_similarity_score == 0: # Prevent division by 0 error
    total_similarity_score = 1
  md_similarity_score = total_similarity_score / total_max_similarity_score
  overall_score = (rouge2.fmeasure + rougeL.fmeasure + 2 * md_similarity_score) / 4

  print(f"Rouge 2: {rouge2.fmeasure * 100}%")
  print(f"Rouge L: {rougeL.fmeasure * 100}%")
  print(f"MD Similarity: {md_similarity_score * 100}%")
  print(f"Overall: {overall_score * 100}%")

  return overall_score * 100, rouge2.fmeasure * 100, rougeL.fmeasure * 100, md_similarity_score * 100

if __name__ == "__main__":
  dataframe = pd.DataFrame({
    'generated': ['This is a test.', 'What a nice day.'],
    'expected': ['This is a test.', 'What a nice day.'],
    'generated_md': ['(MD0)This is (MD1)a test(MD1).(MD0)', '(MD0)What a nice day.(MD0)'],
    'original_md': ['(MD0)Can you (MD1)believe this?(MD1) This was actually just a (MD1)test(MD1).(MD0)', '(MD0)What a beautiful day.(MD0)']
  })
  print(get_words_similarity("This is a test.", "This is a second test."))
  print(get_words_similarity("This is a test.", "This is a test."))
  print(get_words_similarity("this is a cool dog", "want to eat food"))
  print(get_words_similarity("believe this?", "believe this"))