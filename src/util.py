import re
from bs4 import BeautifulSoup
import markdown
import markdownify

md = markdown.Markdown(extensions=['pymdownx.mark', 'pymdownx.tilde', 'pymdownx.emoji', 'pymdownx.extra'])

def mdEncode(text: str):
  html = md.convert(text)
  soup = BeautifulSoup(html, "html.parser")
  mdLookup = []
  result = ""
  root = soup.find('p')
  if root is None:
    root = soup
  for element in root.children:
    if element.name:
      mdIndex = len(mdLookup)
      tag = element.name
      mdInfo = {'tag': tag, 'attrs': element.attrs}
      if mdInfo in mdLookup:
        mdIndex = mdLookup.index(mdInfo)
      else:
        mdLookup.append(mdInfo)
      result += f"(MD{mdIndex}){element.text}(MD{mdIndex})"
    else:
      result += element
  return result, mdLookup

def mdDecode(text: str, mdLookup):
  soup = BeautifulSoup()
  textMdTags = re.split(r"(\(MD\d+\)+)", text)
  openedTag = False
  mdIndex = 0
  for textOrMdTag in textMdTags:
    if re.search(r"\(MD\d+\)", textOrMdTag):
      openedTag = not openedTag
      mdIndex = int(re.findall(r"\d+", textOrMdTag)[0])
    else:
      if openedTag:
        mdInfo = mdLookup[mdIndex]
        tag = soup.new_tag(f"{mdInfo['tag']}", **mdInfo['attrs'])
        tag.string = textOrMdTag
        if tag.name == "mark":
          tag = f"=={textOrMdTag}=="
        soup.append(tag)
      else:
        soup.append(textOrMdTag)
  return markdownify.markdownify(str(soup))

def mdRemove(text: str):
  text = re.sub(r"\(MD\d+\)", "", text)
  text = re.sub(r" +", " ", text).strip()
  text = re.sub(r'\s([?.!"](?:\s|$))', r"\1", text)
  return text