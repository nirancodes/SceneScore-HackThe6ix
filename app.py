import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

import google.generativeai as genai 
from lyricsgenius import Genius
import re 
from sentence_transformers import SentenceTransformer, util 
from dotenv import load_dotenv
import os 

load_dotenv()

genius_token = os.getenv("GENIUS_TOKEN")
genius = Genius(genius_token)

def cleanTitleForLyricsSearch(title): 
    title = re.sub(r"[-–]\s*From\s+[\"'].*?[\"']", "", title)
    title = re.sub(r"\(From\s+[\"'].*?[\"']\)", "", title)
    title = re.sub(r"\[.*?\]|\(.*?\)", "", title)
    return title.strip(" -–").strip()

# Safer to use env variable instead of hardcoding
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

@st.cache_data(show_spinner=False)
def GeminiSongDescription(song, lyrics=None): 
    prompt = f"""
    You are a music critic. Based on the following song metadata and lyrics, generate only one sentence describing the song's meaning, scene, vibe mood and emotional context. 
    - Title: {song['name']}
    - Artist: {song['artist']}
    - Album: {song['album']}
    - Track Popularity: {song['track_popularity']}
    - Artist Popularity: {song['artist_popularity']}
    - Genres: {', '.join(song['genres']) if song['genres'] else 'Unknown'}

    Lyrics: 
    {lyrics if lyrics else 'Lyrics not available'}

    
    """
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    response = model.generate_content(prompt)
    return response.text.strip()


# Streamlit setup
st.set_page_config(page_title="SceneScore", layout="centered")

# Custom CSS
st.markdown("""
<style>
.gradient-text {
  font-size: 3em;
  font-weight: 800;
  background: linear-gradient(to right, #f58529, #dd2a7b, #8134af, #515bd4);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  text-align: center;
  margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='gradient-text'>SceneScore</div>", unsafe_allow_html=True)

# Upload photo
uploaded_file = st.file_uploader("📷 Upload your photo", type=["jpg", "png", "jpeg"])

@st.cache_resource(show_spinner=False)
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

@st.cache_resource
def load_embedding_model(): 
    return SentenceTransformer("all-mpnet-base-v2")
# Spotify auth
@st.cache_resource
def load_spotify_client():
    creds = SpotifyClientCredentials(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET")
    )
    return spotipy.Spotify(client_credentials_manager=creds)

sp = load_spotify_client()

def get_playlist_tracks(playlist_link):
    playlist_URI = playlist_link.split("/")[-1].split("?")[0]
    tracks = sp.playlist_tracks(playlist_URI)["items"]

    song_data = []
    for track_item in tracks:
        track = track_item["track"]
        artist_uri = track["artists"][0]["uri"]
        artist_info = sp.artist(artist_uri)

        song_data.append({
            "name": track["name"],
            "artist": track["artists"][0]["name"],
            "album": track["album"]["name"],
            "track_popularity": track["popularity"],
            "artist_popularity": artist_info["popularity"],
            "genres": artist_info["genres"],
        })
    return song_data

@st.cache_data(show_spinner=False)
def GeminiJustifyMatch(caption, top_song, runner_ups):
    runner_up_text = "".join([
    f"- {s['name']} by {s['artist']}, Score: {s['score']:.3f}, Description: {s['description']}\n"
    for s in runner_ups
    ])
    prompt = f"""
    You're a music match expert. A system tried to match the following image description with songs:

    Image Description: "{caption}"

    Top Match:
    - Title: {top_song['name']} by {top_song['artist']}
    - Description: {top_song['description']}
    - Score: {top_song['score']:.3f}

    Other Candidates:
    {runner_up_text}

    In 3 sentences, explain why the top match was a better fit for the image scene than the other candidates, focusing on mood, emotion, or theme overlap.
        """
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    response = model.generate_content(prompt)
    return response.text.strip()

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("⏳ Generating image caption... please wait."):
        processor, model, device = load_blip()
        inputs = processor(image, return_tensors="pt").to(device)
        generated_ids = model.generate(**inputs)
        caption = processor.decode(generated_ids[0], skip_special_tokens=True)
        embedder = load_embedding_model()
        caption_embedding = embedder.encode(caption, convert_to_tensor=True)


    st.success(f"📝 Description: **{caption}**")

    # Now load and show tracks
    st.markdown("---")
    st.markdown("🎵 **Your Playlist's Songs & Metadata**")
    

    playlist_link = "https://open.spotify.com/embed/playlist/4s4PlzHgQBu9AuCplXIIVQ"
    songs = get_playlist_tracks(playlist_link)


    allSongScores = []

    for i, song in enumerate(songs):
        with st.expander(f"{i+1}. {song['name']} by {song['artist']}"):
            st.markdown(f"- **Album**: {song['album']}")
            st.markdown(f"- **Popularity**: {song['track_popularity']}")
            st.markdown(f"- **Artist Popularity**: {song['artist_popularity']}")
            st.markdown(f"- **Genres**: {', '.join(song['genres']) if song['genres'] else 'N/A'}")
            
            lyrics = None 
            raw_title=song['name']
            clean_title = cleanTitleForLyricsSearch(raw_title)
            try: 
                genius_song = genius.search_song(clean_title, song['artist'])
                if genius_song and genius_song.lyrics:
                    lyrics = genius_song.lyrics
            except Exception as e:
                st.warning("⚠️ Could not fetch lyrics for this song.")
                
            with st.spinner("🔍 Analyzing song vibe with Gemini..."):
                songDescription= GeminiSongDescription(song, lyrics)
                song_embedding = embedder.encode(songDescription, convert_to_tensor=True)
                similarity_score = util.cos_sim(caption_embedding, song_embedding).item()
                st.markdown(f"🎯 **Match Score with Image Scene**: `{similarity_score:.3f}`")
            st.markdown("🧠 **Gemini Description of Vibe & Emotion:**")
            st.markdown(f"> {songDescription}")

             
            allSongScores.append({
                "name": song["name"],
                "artist": song["artist"],
                "description": songDescription, 
                "score": similarity_score, 
                "youtube_query": f"{song['name']} {song['artist']}",
            })

    sorted_songs = sorted(allSongScores, key=lambda x: x['score'], reverse=True)
    top_song = sorted_songs[0]
    runner_ups = sorted_songs[1:3]



    st.header("🔝 Best Song Match for Your Image Scene Compared to the Top 3")

    st.subheader(f"{top_song['name']} by {top_song['artist']} (Score: {top_song['score']:.3f})")

    with st.spinner("🔍 Justifying top match over others..."):
        comparison_justification = GeminiJustifyMatch(caption, top_song, runner_ups)

    st.markdown(f"📝 **Why This Song Was Chosen Over Others:** {comparison_justification}")

    youtube_query = top_song['youtube_query']
    youtube_url = f"https://www.youtube.com/results?search_query={youtube_query.replace(' ', '+')}"
    st.markdown(f"[🎥 Watch on YouTube]({youtube_url})")





#__________________________________________________________________________________________________
# 

#---------------------------------------------------------------------------------------------------------------
# image_caption = "A moody landscape with dark clouds and a lone figure walking across the field."

# songs = [
#     {
#         "title": "Haunted Heart",
#         "description": "A melancholic ballad with slow piano melodies and ethereal vocals evoking solitude and longing.",
#     },{
#         "title": "Sunset Fade",
#         "description": "Ambient guitar layers over a soft beat, capturing the feeling of watching the sun sink into the ocean.",
#     },{
#         "title": "Echoes in Snow",
#         "description": "A minimal electronic piece with icy synths and distant chimes, evoking the silence and purity of a snowy night.",
#     },{
#         "title": "Urban Mirage",
#         "description": "Fast-paced lo-fi beats layered with cityscape samples and reverb, simulating the disorienting beauty of nightlife in a neon-lit metropolis.",
#     },{
#         "title": "Golden Reverie",
#         "description": "Warm acoustic strumming paired with soft humming, channeling the nostalgia of summer memories and golden-hour light.",
#     }
# ]


# import torch 
# import pandas as pd

# model = SentenceTransformer("all-mpnet-base-v2")

# imageVector = model.encode(image_caption, convert_to_tensor=True)

# for song in songs: 
#     song['embedding'] = model.encode(song['description'], convert_to_tensor=True)
#     similarity = util.pytorch_cos_sim(imageVector, song['embedding']).item()
#     song['score'] = similarity

# songs = sorted(songs, key=lambda x: x['score'], reverse=True)
# df=pd.DataFrame(songs)

# print(f"Image Caption: {image_caption}\n")
# print("Top 3 songs ranked by similarity:")

# # Streamlit UI
# st.title("🎵 Image to Song Similarity")
# st.subheader("🖼️ Image Caption")
# st.write(f"**{image_caption}**")

# st.subheader("📊 Song Similarity Scores")
# st.bar_chart(df.set_index('title')['score'])

# # Optional: show top matches
# st.subheader("🔝 Top 3 Matches")
# for i, row in df.head(3).iterrows():
#     st.markdown(f"**{i+1}. {row['title']}** - Similarity: `{row['score']:.4f}`")
#     st.caption(row['description'])