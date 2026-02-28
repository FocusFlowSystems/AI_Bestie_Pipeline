# AI Bestie Content Pipeline: AI-Driven Long-to-Short Content Engine

AI Bestie Content Pipeline is an automated, nightly batch-processing pipeline that transforms raw, long-form video recordings into fully polished, short-form social media assets. 

It leverages local GPU transcription (Whisper), dynamic Karaoke-style subtitle generation, precise FFmpeg video slicing, and a highly advanced "Virtual Focus Group" driven by Google Gemini's **Mendelsohn Kernel Standard (MKS)**.

---

## 🚀 The Architecture

1. **Ingestion & Audio Extraction**: The pipeline automatically detects new `.mp4` or `.mkv` files in `01_Raw_Recordings`, extracts a highly compressed 32k Mono MP3, and moves the assets to a staging directory.
2. **Local Transcription**: Uses OpenAI's Whisper model (running locally for zero API cost) to generate highly accurate, word-level JSON transcripts.
3. **Karaoke Subtitle Engine**: Converts the Whisper JSON into dynamic, 2-line `.ass` subtitle files, highlighting the active speaker's exact word in cyan and bounding the text to ensure perfect vertical video readability.
4. **MKS Virtual Focus Group (AI Director)**: The pipeline reads 5 distinct audience archetype "Personas" from the `personas/` directory. Gemini 3 Flash spins up independent instances for each persona to review the transcript, flag "Slop", verify safety, and identify the perfect 60-120 second video slices based on their unique Axioms and Logic Gates.
5. **Clip Unification & Slicing**: If multiple personas identify the same segment, the script merges the overlap. It then uses FFmpeg's "Jump and Decode" method to extract millisecond-perfect video clips.
6. **Voice Synthesizer**: The raw critique from the 5 MKS personas is passed to a final LLM Synthesis step. This generates a cohesive social media caption in the creator's authentic voice, overriding the AI's "auditor" tone.
7. **Cloud Sync**: All final assets—including the raw video, extracted clips, Karaoke subtitles, and text files containing YouTube metadata and Infographic prompts—are bundled cleanly into a timestamped folder and synced directly to a mounted Google Drive.

---

## 👥 The Mendelsohn Kernel Standard (MKS)

AI Bestie abandons generic "LLM prompts" in favor of the **Mendelsohn Kernel Standard**. By dropping JSON profiles into the `personas/` directory, the pipeline simulates a diverse Virtual Focus Group.

Included Archetypes:
* **The Practical Ops** (`KERNEL_PRACTICAL_OPS_V1`): Scrutinizes content for ROI, actionable systems, and efficiency.
* **The Strategy Architect** (`KERNEL_STRATEGY_ARCHITECT_V1`): Evaluates for deep mental models, corporate governance, and leadership alignment.
* **The Neuro Ally** (`KERNEL_NEURO_ALLY_V1`): Flags dense "Walls of Text" and ensures the content is low-friction and accessible via the lens of executive function.
* **The Guardian Guide** (`KERNEL_GUARDIAN_GUIDE_V1`): Audits the script for beginner accessibility, privacy, and safety for digital natives.
* **The Slop Hunter** (`KERNEL_SLOP_HUNTER_V1`): The pipeline's rigorous Quality Control. Flags generic platitudes, "game-changer" vocabulary, and hallucinations.

---

## 🛠️ Setup & Installation

### 1. System Requirements
- Linux Environment
- Python 3.10+
- `ffmpeg` installed on the system path

### 2. Environment Configuration
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY="YOUR_GOOGLE_AI_STUDIO_KEY"
DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/your/url"
GOOGLE_DRIVE_PATH="/path/to/your/mounted/google_drive/AI_Bestie_Drop"
```

### 3. Dependencies
```bash
pip install openai-whisper python-dotenv google-genai requests
```

### 4. Running the Pipeline
Place raw video files in `01_Raw_Recordings`.

**Manual Run:**
```bash
python3 ai_bestie_pipeline.py
```

**Automated Nightly Run (Crontab):**
```bash
crontab -e
# Add the following line to run every night at 9:40 PM:
40 21 * * * /usr/bin/python3 /absolute/path/to/AI_Bestie/ai_bestie_pipeline.py
```

---

## 📂 Output Deliverables

For every video processed, AI Bestie generates a bundle in `03_Final_Assets` (and syncs it to Google Drive) containing:
- `RAW_[VideoName].mp4`
- `CAPTIONED_[VideoName].mp4`
- The generated Word-Level JSON transcript and `.ass` subtitle file.
- `SHORT_[Title].mp4` (x The number of unique clips identified).
- `social_descriptions.txt`: Contains your full-length YouTube Title & Description, raw MKS Persona feedback, and synthesized video captions.
- `text_posts_with_infographics.txt`: Contains extracted, standalone text posts for LinkedIn alongside detailed AI Image Generator prompts to create accompanying infographics.
