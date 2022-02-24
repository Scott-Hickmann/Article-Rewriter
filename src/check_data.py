import pandas as pd
from util import md, mdDecode, mdEncode

dataframe = pd.read_csv('data/annotated_data.csv', encoding='utf-8')

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

for text in dataframe['source_with_markdown']:
  checkData(text)

for text in dataframe['target_with_markdown']:
  checkData(text)