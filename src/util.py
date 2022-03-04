import re
from bs4 import BeautifulSoup
import markdown
import markdownify

md = markdown.Markdown(extensions=['pymdownx.mark', 'pymdownx.tilde', 'pymdownx.emoji', 'pymdownx.extra'])

stLookup = ["\n\n", "\n"]

def mdEncode(text: str):
  for i, specialToken in enumerate(stLookup):
    text = text.replace(specialToken, f"<st{i}/>")
  html = md.convert(text)
  soup = BeautifulSoup(html, "html.parser")
  mdLookup = []
  result = ""
  root = soup.find('p')
  if root is None:
    root = soup

  def mdEncodeRec(element):
    if element.name is None:
      return element
    if re.search(r"st\d+", element.name):
      stIndex = int(re.findall(r"\d+", element.name)[0])
      return f"(ST{stIndex})"
    mdIndex = len(mdLookup)
    tag = element.name
    mdInfo = {'tag': tag, 'attrs': element.attrs}
    if mdInfo in mdLookup:
      mdIndex = mdLookup.index(mdInfo)
    else:
      mdLookup.append(mdInfo)
    localResult = ""
    for child in element.children:
      localResult += mdEncodeRec(child)
    return f"(MD{mdIndex}){localResult}(MD{mdIndex})"

  for element in root.children:
    if element.name:
      result += mdEncodeRec(element)
    else:
      result += element
  return result, mdLookup

def mdDecode(text: str, mdLookup):
  textMdTags = re.split(r"(\(MD\d+\))", text)
  html = ""
  openedTags = []
  links = []
  linkReplacer = "http://LINK.link"
  for textOrMdTag in textMdTags:
    if re.search(r"\(MD\d+\)", textOrMdTag):
      mdIndex = int(re.findall(r"\d+", textOrMdTag)[0])
      mdInfo = mdLookup[mdIndex]
      tag = mdInfo['tag']
      if (len(openedTags) > 0 and openedTags[-1] == tag):
        openedTags.pop()
        html += "==" if tag == "mark" else f"</{tag}>"
      else:
        openedTags.append(tag)
        if tag == "mark": 
          html += "=="
        else:
          html += f"<{tag}"
          for key, value in mdInfo['attrs'].items():
            if key == "href":
              links.append(value)
              value = linkReplacer
            html += f" {key}=\"{value}\""
          html += ">"
    else:
      html += textOrMdTag
  
  markdown = markdownify.markdownify(html)
  markdownOrLinks = re.split(r"(http://LINK.link)", markdown)
  result = ""
  linkIndex = 0
  for markdownOrLink in markdownOrLinks:
    if markdownOrLink == linkReplacer:
      result += links[linkIndex]
      linkIndex += 1
    else:
      result += markdownOrLink


  for i, specialToken in enumerate(stLookup):
    result = result.replace(f"(ST{i})", specialToken)
  return result

def mdRemove(text: str):
  text = re.sub(r"\(MD\d+\)", "", text)
  text = re.sub(r" +", " ", text).strip()
  text = re.sub(r'\s([?.!"](?:\s|$))', r"\1", text)
  for i, specialToken in enumerate(stLookup):
    text = text.replace(f"(ST{i})", " ")
  return text