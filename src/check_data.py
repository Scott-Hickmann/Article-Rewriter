import pandas as pd
from util import md, mdDecode, mdEncode

def checkData(text: str):
  mdText, mdLookup = mdEncode(text)
  mdDecodedText = mdDecode(mdText, mdLookup)
  html = md.convert(text)
  mdDecodedHtml = md.convert(mdDecodedText)
  if html != mdDecodedHtml:
    print("Error:")
    print(f"Expected: {text}")
    print(f"Received: {mdDecodedText}")
    print()

dataframe = pd.read_csv('data/paraphraser/annotated_data.csv', encoding='utf-8')

for text in dataframe['source_with_markdown']:
  checkData(text)

for text in dataframe['target_with_markdown']:
  checkData(text)

dataframe = pd.read_csv('data/summarizer/annotated_data.csv', encoding='utf-8')

for text in dataframe['source_with_markdown']:
  checkData(text)

for text in dataframe['target_with_markdown']:
  checkData(text)