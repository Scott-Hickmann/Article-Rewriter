import re
from unicodedata import name
from bs4 import BeautifulSoup
import markdown
import markdownify

md = markdown.Markdown(extensions=['sane_lists', 'pymdownx.mark', 'pymdownx.tilde', 'pymdownx.emoji', 'pymdownx.extra'])

def mdEncode(text: str, existingMdLookup=None):
  html = md.convert(text)
  root = BeautifulSoup(html, "html.parser")
  mdLookup = existingMdLookup if existingMdLookup else []
  result = ""

  def mdEncodeRec(element):
    if element.name is None:
      return element
    if re.search(r"st\d+", element.name):
      stIndex = int(re.findall(r"\d+", element.name)[0])
      return f"(ST{stIndex})"
    tag = element.name
    mdInfo = {'tag': tag, 'attrs': element.attrs}
    if mdInfo in mdLookup:
      mdIndex = mdLookup.index(mdInfo)
    else:
      mdIndex = len(mdLookup)
      mdLookup.append(mdInfo)
    mdIndex = mdLookup.index(mdInfo)
    localResult = ""
    for child in element.children:
      localResult += mdEncodeRec(child)
    return f"(MD{mdIndex}){localResult}(MD{mdIndex})"

  for element in root.children:
    if element.name:
      result += mdEncodeRec(element)
    else:
      result += element

  result = result.replace("\n", "")
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

  return result.strip()

def mdRemove(text: str):
  text = re.sub(r"\(MD\d+\)", "", text)
  text = re.sub(r" +", " ", text).strip()
  text = re.sub(r'\s([?.!"](?:\s|$))', r"\1", text)
  return text

if __name__ == "__main__":
  original = "**Hello** _world_\n\nHow are\nyou\n\ntoday"
  originalMdEncoded, originalMdLookup = mdEncode(original)
  print(original)
  print(originalMdEncoded)
  originalMdDecoded = mdDecode(originalMdEncoded, originalMdLookup)
  print(originalMdDecoded)
  variant1 = "_Hello_ **world**\n\nHow are\nyou\n\ntoday"
  variant1MdEncoded, variant1MdLookup = mdEncode(variant1)
  print(variant1)
  print(variant1MdEncoded)
  if variant1MdEncoded != originalMdEncoded:
    raise Exception("Encoded variant does not match original")
  if variant1MdLookup == originalMdLookup:
    print(variant1MdLookup)
    print(originalMdLookup)
    raise Exception("Encoded variant lookup should not match original")
  variant2MdEncoded, variant2MdLookup = mdEncode(variant1, originalMdLookup)
  print(variant1)
  print(variant2MdEncoded)
  if variant2MdEncoded == originalMdEncoded:
    raise Exception("Encoded variant should not match original")
  if variant2MdLookup != originalMdLookup:
    raise Exception("Encoded variant lookup does not match original")
  print(mdDecode(*mdEncode("""Here is a pair of old iOS 6 settings — “Do Not Disturb” and “Notifications”. Look how many light effects are going on with them.

*   The top lip of the inset control panel casts a small shadow
*   The “ON” slider track is also immediately set in a bit
*   The “ON” slider track is concave and the bottom reflects more light
*   The icons are set _out_ a bit. See the bright border around the top of them? This represents a surface perpendicular to the light source, hence receiving a lot of light, hence bouncing a lot of light into your eyes.
*   The divider notch is shadowed where angled away from the sun and vice versa""")))
  print(mdDecode(mdEncode("If you feel the itch to pull out your phone in a break in the conversation, ‘silence’ it. If you want to truly cleanse, this step is unbreakable. You can influence your friends’ behaviors by playfully shaming them when they pull out their phones unnecessarily.")[0], [{'tag': 'ol', 'attrs': {}}, {'tag': 'li', 'attrs': {}}, {'tag': 'strong', 'attrs': {}}, {'tag': 'em', 'attrs': {}}, {'tag': 'a', 'attrs': {'href': 'https://medium.com/'}}, {'tag': 'a', 'attrs': {'href': 'http://i2.kym-cdn.com/photos/images/original/000/133/088/2320c993_92f3_b20b.jpg'}}]))