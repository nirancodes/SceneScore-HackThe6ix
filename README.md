# SceneScore — Perfect Song Matches for Your Photos, Finally

Imagine this: You’re scrolling endlessly through Spotify, trying to find that perfect song for your Instagram story or photo. Hours wasted, mood lost. Until now.  
SceneScore matches your personal Spotify playlist songs with the emotional vibe of your photo — no more guesswork, just seamless, personalized soundtrack curation.

[![Watch the demo](https://i.vimeocdn.com/video/your-thumbnail-image.jpg)](https://vimeo.com/1102849578)

---

## 🎤 Why Lyrics Were the Next Best Signal

Spotify stopped giving devs access to detailed audio features and analysis as of Nov 2024 — a big blow to emotion-based matching. So I turned to Genius API to fetch song lyrics.  
Lyrics are emotionally dense, narratively rich, and capture a song’s mood and scene far beyond raw metadata like popularity or genre.

This let me go beyond what a song is, to what it feels like:  
lyrics → natural language → embedding → emotion vector.

---

## 📐 Why Cosine Similarity Was the Right Fit

Once both the image caption (via BLIP) and song summary (via Gemini + lyrics) became embeddings, I compared them using cosine similarity.  
It measures directional alignment (angle) between vectors — perfect for matching concepts like “lonely field” and “haunting melody.”

- Scale-invariant: longer descriptions don’t overpower shorter ones.
- Semantically meaningful: higher cosine means stronger emotional/mood alignment.

This gave me a quantitative, interpretable metric to rank songs by emotional resonance with your photo.

---

## 🔒 Why I Chose Client Credentials Flow (Not User Auth)

For MVP speed and simplicity:  
I used Client Credentials Flow — authenticates the app, not the user.

- This limits access to public playlists only.
- Avoided the complexity of OAuth token refresh, UI redirects, and user login friction.

Tradeoff? No access yet to Liked Songs, Discover Weekly, or listening history. Next step: upgrade to Authorization Code Flow for deeper personalization.

---

## 🤖 Why I Used BLIP (Not Gemini Vision) for Image Captioning

Three reasons:

- **Cost:** Gemini Vision is metered; BLIP is open-source and runs locally.
- **Flexibility:** BLIP’s architecture allows future fine-tuning or model swaps.
- **Embedding Compatibility:** BLIP produces natural language captions that smoothly feed into SentenceTransformer embeddings.

BLIP is the perfect bridge: image → caption → embedding for SceneScore’s pipeline.

---

## 🔍 How Gemini API Powers Explainability & Semantics

Gemini isn’t just a black box summary tool here — I use it extensively for user trust and clarity:

- It generates vivid, human-readable descriptions of each song’s vibe, based on metadata + lyrics.
- It explains why the top matched song fits better emotionally than other candidates.

This lets users see how and why the AI matches their moment, not just the final score.  
**SceneScore = interpretable AI, not a magic black box.**

---

## 🌀 Full System Flow

1. User uploads a photo  
2. BLIP generates a textual caption  
3. SentenceTransformer encodes caption → embedding  
4. User inputs a Spotify playlist link  

For each track:  
- Fetch metadata (via Spotipy)  
- Clean title and fetch lyrics (via Genius API)  
- Generate semantic song description (via Gemini)  
- Embed the description  
- Compute cosine similarity with image embedding  
- Rank songs by similarity  
- Gemini explains the top match  
- User previews the match, can watch on YouTube  

---

## 🌍 Real-World Use Cases

- Photographers & filmmakers curating soundtrack moodboards  
- Everyday users creating “movie scene” moments from memories  
- Content creators finding music that feels like their visuals  
- Therapists & journaling apps exploring mood with music & imagery  

---

## 🎯 Core Idea

SceneScore defines “perfect” as personalized.  
Most AI tools push popular content. SceneScore recommends your songs — the ones you already love — matched emotionally to your moments.  
Not guessing what’s “good,” but inferring what’s emotionally resonant via powerful vector space reasoning.

---

## Technical Difficulty & Judging Criteria

### Technical Difficulty

- API hurdles: Spotify’s Nov 2024 restriction on audio features forced a pivot to Genius lyrics — an unstructured, noisy data source — requiring robust parsing, cleaning, and semantic summarization.
- Multi-API orchestration: Integrated Spotify, Genius, Gemini, and HuggingFace BLIP seamlessly in a performant Streamlit app with caching, rate-limit handling, and fallbacks.
- Embedding consistency: Balanced embeddings from different domains (images, text, lyrics, metadata) using SentenceTransformer to ensure meaningful similarity scoring.
- Explainability layer: Built natural-language justifications with Gemini to translate opaque similarity scores into user-trusted explanations — elevating the AI from black-box to interpretable.
- MVP tradeoffs: Chose Client Credentials OAuth for rapid MVP launch, accepting data limitations while architecting for future user-auth flows.

### Uniqueness

- Unlike generic mood or sentiment apps, SceneScore matches your Spotify library, grounding recommendations in personal taste and actual listening history.
- Combines state-of-the-art image captioning, semantic embeddings, and generative AI explainability — a novel fusion rarely seen in hackathons or demos.
- Emotion-driven music curation tied directly to user-generated visual content, redefining “personalized soundtrack” beyond mere popularity metrics.

### Design & User Experience

- Simple, elegant Streamlit UI with fast feedback loops.
- Clear presentation of metadata, song vibe summaries, and match scores.
- User-facing explanations increase trust and understanding.
- Integrated YouTube previews close the discovery loop.

### Completeness

- Fully functional end-to-end pipeline from image upload to song recommendation and explanation.
- Robust error handling for API failures or missing lyrics.
- Cached resources and optimized calls for responsive interaction.
- Laid groundwork for richer personalization and expanded music discovery.

---

## Next Steps & Future Improvements

- **Upgrade to full user authentication:** Move from Client Credentials to Authorization Code Flow to access private playlists, Liked Songs, and richer listening history — unlocking deeper personalization and context-aware recommendations.
- **Broaden platform integration:** Support playlists from other services (Apple Music, YouTube Music) and direct user uploads to grow the user base.
- **User feedback loop:** Collect user ratings on matches to continuously improve the recommendation algorithms via reinforcement learning or fine-tuning.
