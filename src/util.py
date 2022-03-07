import re
from unicodedata import name
from bs4 import BeautifulSoup
import markdown
import markdownify

md = markdown.Markdown(extensions=['sane_lists', 'pymdownx.mark', 'pymdownx.tilde', 'pymdownx.emoji', 'pymdownx.extra'])

def mdEncode(text: str, existingMdLookup=None):
  mdLookup = existingMdLookup if existingMdLookup else []
  html = md.convert(text.replace("\n\n", "DOUBLE_NEW_LINE").replace("\n", "<br/>").replace("DOUBLE_NEW_LINE", "\n\n"))
  root = BeautifulSoup(html, "html.parser")
  for link in root.find_all("a"):
    if link["href"] in mdLookup:
      mdIndex = mdLookup.index(link["href"])
    else:
      mdIndex = len(mdLookup)
      mdLookup.append(link["href"])
    link["href"] = f"LINK{mdIndex}"
  return str(root).replace("\n", ""), mdLookup

def mdDecode(text: str, mdLookup):
  root = BeautifulSoup(text, "html.parser")
  for link in root.find_all("a"):
    mdIndex = int(re.findall(r"\d+", link["href"])[0])
    link["href"] = mdLookup[mdIndex]
  return markdownify.markdownify(str(root)).replace("  \n", "\n").strip()

def mdRemove(text: str):
  root = BeautifulSoup(text.replace("<br/>", "\n").replace("</p>", "</p>\n").strip(), "html.parser")
  return root.text

if __name__ == "__main__":
  def pprint(text: str):
    print(str(text).replace("\n", "\\n"))

  original = "**Hello** _world_\n\nHow are\nyou\n\ntoday from [Google](https://google.com)."
  originalMdEncoded, originalMdLookup = mdEncode(original)
  pprint(mdRemove(originalMdEncoded))
  pprint(original)
  pprint(originalMdEncoded)
  originalMdDecoded = mdDecode(originalMdEncoded, originalMdLookup)
  pprint(originalMdDecoded)
  variant1 = "_Hello_ **world**\n\nHow are\nyou\n\ntoday"
  variant1MdEncoded, variant1MdLookup = mdEncode(variant1)
  pprint(variant1)
  pprint(variant1MdEncoded)
  variant2MdEncoded, variant2MdLookup = mdEncode(variant1, originalMdLookup)
  pprint(variant1)
  pprint(variant2MdEncoded)
  print(mdDecode(*mdEncode("""Here is a pair of old iOS 6 settings — “Do Not Disturb” and “Notifications”. Look how many light effects are going on with them.

*   The top lip of the inset control panel casts a small shadow
*   The “ON” slider track is also immediately set in a bit
*   The “ON” slider track is concave and the bottom reflects more light
*   The icons are set _out_ a bit. See the bright border around the top of them? This represents a surface perpendicular to the light source, hence receiving a lot of light, hence bouncing a lot of light into your eyes.
*   The divider notch is shadowed where angled away from the sun and vice versa""")))
  print(mdDecode(mdEncode("If you feel the itch to pull out your phone in a break in the conversation, ‘silence’ it. If you want to truly cleanse, this step is unbreakable. You can influence your friends’ behaviors by playfully shaming them when they pull out their phones unnecessarily.")[0], [{'tag': 'ol', 'attrs': {}}, {'tag': 'li', 'attrs': {}}, {'tag': 'strong', 'attrs': {}}, {'tag': 'em', 'attrs': {}}, {'tag': 'a', 'attrs': {'href': 'https://medium.com/'}}, {'tag': 'a', 'attrs': {'href': 'http://i2.kym-cdn.com/photos/images/original/000/133/088/2320c993_92f3_b20b.jpg'}}]))