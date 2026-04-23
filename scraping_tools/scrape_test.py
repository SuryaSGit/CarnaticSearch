import requests
from bs4 import BeautifulSoup
import json
import re
import unicodedata
import time
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
def normalize(text: str) -> str:
    # lowercase
    text = text.lower()
    
    # remove accents/diacritics
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    
    # normalize common Carnatic transliteration quirks
    replacements = {
        "aa": "a",
        "ee": "i",
        "oo": "u",
        "bh": "b",
        "dh": "d",
        "th": "t",
        "sh": "s",
        "ṣ": "s",
        "ṅ": "n",
        "ñ": "n"
    }
    
    for k, v in replacements.items():
        text = text.replace(k, v)
    
    # remove anything not letters or spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
url = 'https://www.karnatik.com/lyricstext.shtml'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
}
response = requests.get(url, headers=headers, verify=False)
links = []

if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')
    
    i = 0
    rows = soup.select('li')
    for li in rows:
        if(i < 7926):
            i = i + 1
            continue
        if(i > 8030):
            break
        link = li.find('a')
        if link and 'href' in link.attrs:
            href = link['href']
            text = link.get_text(strip=True)
            i += 1
            links.append("https://www.karnatik.com/" + href)

else:
    print(f"Failed to retrieve page. Status code: {response.status_code}")
print("--------------LINKS SCRAPED---------------")
lyrics = []
errors = []
with open('progress.json', 'r') as file:
    to_upload = json.load(file)
id_number = 7925
for link in links:
    data = {
        "Song Name": "",
        "Raagam": "",
        "Taalam": "",
        "Composer": "",
        "Language": "",
        "Lyrics": "",
        "Id": id_number
    }
    id_number = id_number + 1
    time.sleep(2)
    response = requests.get(link, headers=headers, verify=False)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        content = soup.get_text("\n", strip=True)
        lines = content.split('\n')
        for i in range(len(lines)):
            line = lines[i]
            if(line.startswith("Song:")):
                while(not lines[i+1].startswith("raagam")):
                    i = i + 1
                    if(i == len(lines)-1):
                        break
                if(i == len(lines)-1):
                    continue
                data["Song Name"] = lines[i]
            if(line.startswith("taaLam:")):
                data["Taalam"] = line.replace("taaLam:", "").strip()
            if(line.startswith("raagam:")):
                if(not lines[i+1].startswith("taaLam:")):
                    data["Raagam"] = lines[i+1]
            if(line.startswith("Language")):
                data["Language"] = line.replace("Language:", "").strip()
                i = i + 1
                while(not lines[i].startswith("Meaning:") and not lines[i].startswith("first") and not lines[i].startswith("Other information:")):
                    if(i == len(lines)-1):
                        break
                    if(lines[i].startswith("pallavi") or lines[i].startswith("Pallavi") or lines[i].startswith("anupallavi") or lines[i].startswith("Anupallavi") or lines[i].startswith("caranam") or lines[i].startswith("Caranam") or lines[i].startswith("caraNam") or lines[i].startswith("CaraNam")):
                        i = i + 1
                        continue
                    data["Lyrics"]+=(normalize(lines[i]))
                    i = i + 1
                if(i == len(lines)-1):
                    continue
            if(line.startswith("Composer:")):
                data["Composer"] = lines[i+1]
    else:
        errors.append(link)
    if(data["Song Name"] == ""):
        errors.append(link)
        continue
    if(data["Lyrics"] == ""):
        errors.append(link)
        continue
    to_upload.append(data)
    with open("progress.json", "w") as f:
        json.dump(to_upload, f)
with open("data.json", "w") as file:
    json.dump(to_upload, file)

with open("errors.txt", "w") as f:
    for error in errors:
        f.write(f"{error}\n")
