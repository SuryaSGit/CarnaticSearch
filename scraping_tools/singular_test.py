import requests
from bs4 import BeautifulSoup

url = 'https://www.karnatik.com/c8930.shtml'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, 'html.parser')

# Target the main container shown in your image
container = soup.find('font', face='tahoma')

data = {
    "Song Name": "",
    "Raagam": "",
    "Taalam": "",
    "Composer": "",
    "Language": "",
    "Lyrics": []
}

if container:
    # 1. Extract Song Name (usually inside the first <b> tag)
    song_tag = container.find('b')
    if song_tag:
        data["Song Name"] = song_tag.get_text(strip=True).replace("Song:", "")

    # 2. Extract Raagam (Look for the link that points to ragas)
    raagam_link = container.find('a', href=lambda x: x and 'rag' in x)
    if raagam_link:
        data["Raagam"] = raagam_link.get_text(strip=True)

    # 3. Extract other metadata and lyrics
    # We iterate through all text strings to find labels
    all_text = list(container.stripped_strings)
    
    capture_lyrics = False
    
    for i, text in enumerate(all_text):
        lower_text = text.lower()
        
        # Metadata logic
        if "taalam:" in lower_text:
            data["Taalam"] = all_text[i+1] if i+1 < len(all_text) else ""
        elif "composer:" in lower_text:
            data["Composer"] = all_text[i+1] if i+1 < len(all_text) else ""
        elif "language:" in lower_text:
            data["Language"] = all_text[i+1] if i+1 < len(all_text) else ""
            
        # Lyrics capture logic
        if any(keyword in lower_text for keyword in ["pallavi", "anupallavi", "caranam", "caraNam"]):
            capture_lyrics = True
        
        if "meaning" in lower_text:
            capture_lyrics = False
            
        if capture_lyrics:
            data["Lyrics"].append(text)

# Clean up results
print(f"SONG: {data['Song Name']}")
print(f"RAAGAM: {data['Raagam']}")
print(f"TAALAM: {data['Taalam']}")
print(f"COMPOSER: {data['Composer']}")
print(f"LANGUAGE: {data['Language']}")
print("\n--- LYRICS ---")
print("\n".join(data["Lyrics"]))