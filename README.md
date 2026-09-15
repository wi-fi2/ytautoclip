# OpenSource Clipping

**Turn a long YouTube video into a batch of short, subtitled, vertical clips — automatically.**

## What this project does

You give it a video URL. It gives you back a folder of finished short clips, ready to post — no editing software needed.

Under the hood it runs one pipeline, end to end:

1. **Downloads** the source video (YouTube, TikTok, Instagram, or Google Drive).
2. **Transcribes** it word-for-word with Whisper.
3. **Reads the transcript with Gemini** (an AI) to find the most interesting, "clippable" moments — and to write a title, description, and tags for each one.
4. **Reframes** the video to vertical (9:16, or whatever ratio you want), following faces around the frame automatically.
5. **Renders** each clip with karaoke-style word-by-word subtitles, an optional 3-second hook intro, background music, and an auto-generated thumbnail.
6. *(Optional)* **Uploads** the finished clips straight to YouTube or Facebook Reels on a schedule.

Everything runs from one command: `python main.py --url "..."`.

## Is this for me?

If you've ever manually cut podcast highlights or YouTube "best moments" clips in a video editor, this replaces that manual work. It's aimed at people who post short-form content (Shorts, Reels, TikTok) regularly and want the boring, repetitive part — cutting, reframing, subtitling — automated.

You don't need to know how to code to use it — just how to run a command in a terminal.

## Quick start (5 minutes)

**You need:**
- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) installed and on your PATH
- A free [Google Gemini API key](https://aistudio.google.com/apikey)
- (Recommended) an NVIDIA GPU — it'll work on CPU too, just slower

**Steps:**

```bash
# 1. Get the code
git clone https://github.com/your-username/opensource-clipping.git
cd opensource-clipping

# 2. Install dependencies
pip install -r requirements.txt
# (or: uv sync — if you use uv)

# 3. Add your API key
cp .env.sample .env
# open .env and paste in GOOGLE_API_KEY=your-key-here

# 4. Run it on a video
python main.py --url "https://youtube.com/watch?v=VIDEO_ID" --clips 5 --ratio 9:16
```

Your finished clips will show up in the `outputs/` folder — video files, subtitles, thumbnails, and a JSON report of what was generated and why.

No GPU? See [Running on Google Colab](#running-on-google-colab) below — it's free and requires no local setup at all.

## What you get in `outputs/`

For each clip, the pipeline writes:
- The rendered `.mp4` (subtitled, reframed, with hook + BGM if enabled)
- An auto-generated thumbnail image
- A metadata file with a title, description, and tags ready to paste into YouTube/TikTok
- `clip_candidates.json` — every moment the AI considered, and why it scored the way it did

## Key features

| Feature | What it means in practice |
|---|---|
| **AI clip selection** | Gemini reads the full transcript and picks the moments most likely to hook viewers — you don't scrub through footage yourself |
| **Face-tracking reframe** | Converting 16:9 to 9:16 usually crops people out of frame; this tracks faces and pans smoothly to keep them centered |
| **Karaoke subtitles** | Word-by-word highlighted captions (the "Alex Hormozi" style), burned in automatically |
| **Hook intros** | Each clip opens with a punchy 3-second teaser to stop the scroll |
| **Auto B-roll** | Pulls relevant stock footage from Pexels to cut away to |
| **Podcast modes** | Split-screen or auto camera-switching for two-or-more-speaker podcast clips, using speaker diarization |
| **AI voice-over** | Turn a clip into an original reaction/commentary video using AI-generated script + free text-to-speech |
| **Auto-upload** | Push finished clips to YouTube or Facebook Reels on a schedule, with full metadata |
| **Batch & story modes** | Process a whole list of videos in one run, or stitch clips from multiple sources into one narrative |
| **Resumable pipeline** | Crashes mid-run don't cost you progress — checkpointing skips work that's already done |

Full CLI reference and every flag: run `python main.py --help`, or see [`docs/`](docs/) for deep-dive guides (e.g. [Story Clip Mode](docs/STORY_CLIP.md)).

## Common examples

```bash
# 7 clips with GPU-based face tracking and a custom font
python main.py --url "URL" --clips 7 --face-detector yolo --font-style STORYTELLER

# Two-speaker podcast, split-screen, vertical
python main.py --url "PODCAST_URL" --ratio "9:16" --split-screen

# Square (1:1) for Instagram feed
python main.py --url "URL" --ratio "1:1" --clips 5

# Pull from TikTok or Instagram instead of YouTube
python main.py --url "TIKTOK_URL" --source tiktok --clips 3
```

## Running on Google Colab

No local GPU? Run the whole pipeline for free in a Colab notebook (`notebooks/Lib_OpenSource_Clipping.ipynb` is a ready-to-use template):

```python
!git clone https://github.com/your-username/opensource-clipping.git .
!pip install -r requirements.txt
```

Set your `GOOGLE_API_KEY` via Colab Secrets, then run `main.py` the same way as locally — see the notebook for the full example cell.

## Web Studio (no terminal needed)

There's also a browser-based dashboard — **Clipping Studio** — that connects to a Kaggle/Colab notebook running as the backend, so you can queue jobs and watch progress without touching a command line. See [`web/`](web/) and `notebooks/Kaggle_Studio_Server.ipynb` to set it up.

## Project layout

```
main.py              entry point — this is what you run
clipping/            the pipeline itself (transcription, AI selection, rendering, uploaders)
clipping/phase1/     monetization scoring, checkpointing, batch mode, dedup, QC
docs/                feature deep-dives (Story Clip Mode, etc.)
web/, notebooks/     browser Studio + Colab/Kaggle notebooks
assets/, custom_fonts/  bundled fonts and BGM used in rendering
```

## Prerequisites reference

- **Python** 3.10+
- **FFmpeg** on PATH
- **Google Gemini API key** (required) — [get one here](https://aistudio.google.com/apikey)
- **Pexels API key** (optional, for B-roll) — [get one here](https://www.pexels.com/api/)
- **HuggingFace token** (optional, for podcast split-screen/camera-switch) — [get one here](https://huggingface.co/settings/tokens)
- **CUDA GPU** recommended, not required

---

For the original project this is based on, see [NaufalRizqullah/opensource-clipping](https://github.com/NaufalRizqullah/opensource-clipping).
