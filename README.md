# AI Bestie Content Pipeline: The Weekly Batch Engine & Headless Publisher

AI Bestie Content Pipeline is an end-to-end, autonomous **Weekly Batch Engine** and **Headless Publishing Node**. It transforms raw long-form video recordings into a full 6-day content calendar, including optimized video slicing, social text synthesis, infographic prompts, audio extraction, and dynamic scheduling.

It leverages local GPU transcription (Whisper), dynamic Karaoke subtitle generation, precise FFmpeg video slicing, a highly advanced "Virtual Focus Group" driven by Google Gemini's **Mendelsohn Kernel Standard (MKS)**, and a Redis-backed headless OAuth architecture for zero-touch publishing.

---

## 🚀 The Architecture

### 1. The Weekly Pipeline (`ai_bestie_pipeline.py`)
Run this on Saturday night. It automatically ingests up to 5 raw `.mp4` or `.mkv` modules from `01_Raw_Recordings` and scaffolds a complete `Weekly_Campaign_[Date]` structure:
- **Audio Extraction & Transcription**: Extracts compressed 32k Mono MP3s and uses local Whisper for highly accurate, word-level JSON transcripts.
- **Karaoke Subtitle Engine**: Generates dynamic, 2-line `.ass` subtitle files, highlighting the active speaker's exact word in cyan, bounding the text for optimal vertical video readability.
- **MKS Virtual Focus Group**: Reads audience archetype "Personas" from `personas/` to analyze transcript safety, flag "Slop", and identify the best 60-120s segments.
- **Viral Clip Selection**: Uses a specialized `VIRAL_CLIP_SELECTOR_V1` persona to filter the focus group's favorites down to exactly 2 highly-retentive clips per day.
- **Daily Routing**: Routes the assets into `Day_1_Monday` through `Day_5_Friday` subdirectories.
- **Human-in-the-Loop 00_Infographic_Drop**: Generates a daily folder with standard text and standalone AI Image Generation prompts. (Drop your Midjourney/Leonardo generated image in here for auto-publishing).
- **The Saturday Deep Dive**: Concat-demuxes the 5 raw horizontal videos into a single masterclass video (`Saturday_Deep_Dive.mp4`) via FFmpeg.
- **Cloud Delivery**: Syncs the entire `Weekly_Campaign` bundle cleanly to Google Drive.

### 2. The Execution Controller (`ai_bestie_publisher.py`)
Driven strictly by Cron, this script replaces manual posting by uploading the generated assets to Facebook Pages and YouTube directly from the server.
- Uses **Redis** as a distributed lock (`SETNX`) to ensure overlapping cron jobs never cause race conditions triggering OAuth key revocation.
- Relies on an initial secure interaction (`auth_init.py`) to bypass the "headless impedance mismatch" of standard browser-based OAuth 2.0.
- Handles graceful failure: automatically isolates unauthorized keys into a quarantined state and routes failed payloads to the `05_DLQ/` (Dead Letter Queue).

---

## 👥 The Mendelsohn Kernel Standard (MKS)

AI Bestie abandons generic "LLM prompts" in favor of the **Mendelsohn Kernel Standard**. By dropping JSON profiles into the `personas/` directory, the pipeline simulates a diverse Virtual Focus Group.

Included Archetypes:
* **The Practical Ops** (`KERNEL_PRACTICAL_OPS_V1`): Scrutinizes content for ROI, actionable systems, and efficiency.
* **The Strategy Architect** (`KERNEL_STRATEGY_ARCHITECT_V1`): Evaluates for deep mental models, corporate governance, and leadership alignment.
* **The Neuro Ally** (`KERNEL_NEURO_ALLY_V1`): Flags dense "Walls of Text" and ensures the content is low-friction and accessible via the lens of executive function.
* **The Guardian Guide** (`KERNEL_GUARDIAN_GUIDE_V1`): Audits the script for beginner accessibility, privacy, and safety for digital natives.
* **The Slop Hunter** (`KERNEL_SLOP_HUNTER_V1`): Rigorous Quality Control. Flags generic platitudes, "game-changer" vocabulary, and hallucinations.
* **The Viral Selector** (`VIRAL_CLIP_SELECTOR_V1`): Secondary filter optimizing the final output for 3-second hooks and algorithmic retention.

---

## 🛠️ Setup & Installation

### 1. System Requirements
- Linux Environment
- Python 3.10+
- `ffmpeg` installed on the system path
- `redis-server` installed and active (`sudo apt install redis-server`)

### 2. Environment Configuration
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY="YOUR_GOOGLE_AI_STUDIO_KEY"
DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/your/url"
GOOGLE_DRIVE_PATH="/path/to/your/mounted/google_drive/AI_Bestie_Drop"
```

### 3. Dependencies
```bash
pip install openai-whisper python-dotenv google-genai requests pydantic httpx flask google-api-python-client google-auth-oauthlib google-auth-httplib2 redis
```

### 4. Headless OAuth Initialization
Before the Publisher can run, you must capture the cryptographically secure tokens. Run this interactively on a secure desktop:
```bash
python3 auth_init.py
# Follow the prompts to authorize YouTube, Facebook, and LinkedIn.
```

---

## 📅 The Execution Schedule (Crontab)

To enable true Zero-Touch continuous publishing, add the following to your server's `crontab -e`:

```bash
# ==========================================
# 1. The Weekly Pipeline Run (Saturday Night)
# ==========================================
# Processes 5 raw videos into the Weekly Campaign folder
00 22 * * 6 /usr/bin/python3 /home/luke/AI_Bestie/ai_bestie_pipeline.py

# ==========================================
# 2. The Execution Controller (Publisher)
# ==========================================
# Mon-Fri Drops
00 08 * * 1-5 /usr/bin/python3 /home/luke/AI_Bestie/ai_bestie_publisher.py --job infographic
00 12 * * 1-5 /usr/bin/python3 /home/luke/AI_Bestie/ai_bestie_publisher.py --job short_1
00 15 * * 1-5 /usr/bin/python3 /home/luke/AI_Bestie/ai_bestie_publisher.py --job short_2
00 18 * * 1-5 /usr/bin/python3 /home/luke/AI_Bestie/ai_bestie_publisher.py --job early_release

# Saturday Deep Dive
00 10 * * 6 /usr/bin/python3 /home/luke/AI_Bestie/ai_bestie_publisher.py --job deep_dive
15 10 * * 6 /usr/bin/python3 /home/luke/AI_Bestie/ai_bestie_publisher.py --job cross_pollination
```
