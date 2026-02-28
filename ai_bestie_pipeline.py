import os
import sys
import time
import json
import logging
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# AI imports handled in functions for robustness
try:
    import whisper
except ImportError:
    pass 

# ==========================================
# CONFIGURATION
# ==========================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_KEY_HERE")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
GOOGLE_DRIVE_PATH = Path(os.getenv("GOOGLE_DRIVE_PATH", "/home/luke/google_drive"))

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "01_Raw_Recordings"
STAGING_DIR = BASE_DIR / "02_Staging_Processing"
FINAL_DIR = BASE_DIR / "03_Final_Assets"
PERSONAS_DIR = BASE_DIR / "personas"

# Ensure all directories exist
def init_env():
    for directory in [RAW_DIR, STAGING_DIR, FINAL_DIR, PERSONAS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASE_DIR / "nightly_pipeline.log")
    ]
)

# ==========================================
# TRANSCRIPTION & TIMING ENGINE
# ==========================================

def extract_audio(video_path: Path, audio_path: Path):
    """
    Extract lightweight mono 16kHz audio. 
    NOTE: Whisper runs locally (zero cost). ONLY the text transcript is 
    sent to Gemini, keeping your token spend extremely low.
    """
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame", "-b:a", "32k", str(audio_path)
    ]
    subprocess.run(cmd, capture_output=True, check=True)

def run_whisper(audio_path: Path) -> Path:
    """Run Whisper using the Python API and output word-level JSON."""
    logging.info(f"Running Whisper Python API on {audio_path.name}")
    
    try:
        import whisper
    except ImportError:
        raise ImportError("Whisper module not found. Please install it: pip install openai-whisper")

    # Load model (base)
    model = whisper.load_model("base")
    
    # Transcribe with word timestamps
    result = model.transcribe(str(audio_path), word_timestamps=True)
    
    # Save to JSON
    json_path = STAGING_DIR / f"{audio_path.stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
        
    return json_path

def format_ass_time(seconds: float) -> str:
    """Format seconds into ASS time format: H:MM:SS.CS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centesimals = int(round((seconds - int(seconds)) * 100))
    if centesimals == 100:
        secs += 1
        centesimals = 0
        if secs == 60:
            minutes += 1
            secs = 0
            if minutes == 60:
                hours += 1
                minutes = 0
    return f"{hours}:{minutes:02d}:{secs:02d}.{centesimals:02d}"

def generate_karaoke_ass(json_path: Path, ass_path: Path):
    """Parse Whisper JSON into dynamic 2-line Karaoke .ass subtitles."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    ass_content = [
        "[Script Info]",
        "Title: Antigravity 2-Line Karaoke",
        "ScriptType: v4.00+",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "WrapStyle: 0",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Default,Arial,64,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,2,80,80,426,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
    ]
    
    all_words = []
    for segment in data.get("segments", []):
        all_words.extend(segment.get("words", []))
        
    # Group words into "pages" to limit visible text (approx 2 lines)
    # 12 words is a safe limit for 2 lines at font size 64
    WORDS_PER_PAGE = 12
    pages = [all_words[i:i + WORDS_PER_PAGE] for i in range(0, len(all_words), WORDS_PER_PAGE)]
    
    for page in pages:
        page_end_time = page[-1]["end"]
        for i, current_word in enumerate(page):
            start = current_word["start"]
            # The caption page stays on screen until the next word in the page
            # OR until the end of the page's last word.
            if i + 1 < len(page):
                end = page[i+1]["start"]
            else:
                end = page_end_time
            
            line_parts = []
            for j, w in enumerate(page):
                text = w["word"].strip()
                if j == i:
                    line_parts.append(r"{\c&H00FFFF&}" + text + r"{\c&HFFFFFF&}")
                else:
                    line_parts.append(text)
                    
            full_text = " ".join(line_parts)
            dialogue = f"Dialogue: 0,{format_ass_time(start)},{format_ass_time(end)},Default,,0,0,0,,{full_text}"
            ass_content.append(dialogue)

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ass_content))

# ==========================================
# AI DIRECTOR & NOTIFICATIONS
# ==========================================

def get_ai_director_clips(transcript: str, persona: dict = None) -> list:
    """Calls Google Gemini to identify engaging clips from the transcript."""
    if GEMINI_API_KEY == "YOUR_KEY_HERE" or not GEMINI_API_KEY:
        logging.warning("Gemini API Key not set. Falling back to mock clips.")
        return [{"start_time": 0.0, "end_time": 60.0, "title": "Main_Hook", "description": "Manual description needed."}]

    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        if persona:
            persona_name = persona.get("kernel_id", "Unknown Persona")
            logging.info(f"Asking {persona_name} for their perspective...")
            prompt = f"""
            Act as the following Mendelsohn Kernel Persona:
            {json.dumps(persona, indent=2)}
            
            Analyze the following transcript and identify EXACTLY 1 highly engaging segment (60-120 seconds) that PERFECTLY aligns with your Identity, Axioms, and Logic Gates.
            If a Logic Gate fires, or if an Axiom suggests a specific angle, heavily bias your selection toward that.
            
            Return ONLY a JSON object with a single key named 'clips'. Its value must be an array containing EXACTLY ONE object.
            The object MUST have these exact keys:
            - start_time (float)
            - end_time (float)
            - title (string)
            - persona_feedback (string: detailed feedback, thoughts, flags raised, and why this works for your audience)
            
            CRITICAL: Use the exact timestamps provided in the brackets [start - end] of the Transcript data. Do NOT hallucinate timestamps.
            
            Transcript:
            {transcript}
            """
        else:
            prompt = f"""
            Identify 4-5 highly engaging segments (60-120 seconds each) from this transcript.
            Return ONLY a JSON object with a single key named 'clips'. Its value must be an array of objects.
            Each object MUST have: start_time (float), end_time (float), title (string), description (social post).
            
            CRITICAL: Use the exact timestamps provided in the brackets [start - end] of the Transcript data. Do NOT hallucinate timestamps.
            
            Transcript:
            {transcript}
            """
        
        response = client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                thinking_config=types.ThinkingConfig(
                    # MINIMAL is exclusive to Gemini 3 Flash for peak speed
                    thinking_level=types.ThinkingLevel.MINIMAL 
                )
            )
        )
        
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        
        data = json.loads(raw_text.strip())
        
        parsed_clips = []
        if isinstance(data, list):
            parsed_clips = data
        elif isinstance(data, dict):
            # Handle case where AI returns the clip object directly instead of a list
            if "start_time" in data:
                parsed_clips = [data]
            else:
                parsed_clips = data.get("clips", [])
                
        valid_clips = []
        for c in parsed_clips:
            if isinstance(c, dict) and "start_time" in c and "end_time" in c and "title" in c:
                valid_clips.append(c)
            else:
                logging.warning(f"AI returned invalid clip object (missing required keys): {c}")
                
        return valid_clips
        
    except Exception as e:
        logging.error(f"Gemini Director Node failed: {e}")
        return []

def synthesize_social_post(clip_title: str, persona_feedback: str) -> str:
    """Takes raw persona feedback and synthesizes a final social media post in the user's voice."""
    if GEMINI_API_KEY == "YOUR_KEY_HERE" or not GEMINI_API_KEY:
        return "Manual description needed."

    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        prompt = f"""
        You are Luke Mendelsohn (a professional tech educator and system architect). 
        You are writing content for a short video clip/text post titled "{clip_title}".
        
        I ran this concept through a Virtual Focus Group, and multiple audience personas provided this feedback:
        
        {persona_feedback}
        
        Your task is two-fold:
        1. Write a highly engaging, professional, but authentic social media caption (1-2 short paragraphs) that unifies the core value of this focus group feedback. Write it entirely in YOUR voice for LinkedIn/Facebook. If the Guardian (KERNEL_GUARDIAN_GUIDE_V1) provided feedback, default to style the caption more closely to their perspective. Do not explicitly mention the personas, logic gates, or the focus group.
        2. Write a highly detailed prompt that I can feed into an AI image generator to create a professional infographic/visual that perfectly accompanies the text post.
        
        Return ONLY a JSON object with three keys:
        - "social_post" (string: the final social media caption for the TEXT post)
        - "video_caption" (string: the final social media caption for the VIDEO clip)
        - "infographic_prompt" (string: the prompt for the AI image generator)
        """
        
        response = client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.MINIMAL 
                )
            )
        )
        
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
            
        return json.loads(raw_text.strip())
    except Exception as e:
        logging.error(f"Synthesizer Node failed: {e}")
        return {"social_post": persona_feedback, "infographic_prompt": "Failed to generate infographic prompt."}

def synthesize_long_form_metadata(transcript: str) -> dict:
    """Uses Gemini to generate a Title and YouTube description for the full-length video."""
    if GEMINI_API_KEY == "YOUR_KEY_HERE" or not GEMINI_API_KEY:
        return {"title": "Full Video Title", "description": "Full Video Description"}

    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        prompt = f"""
        You are Luke Mendelsohn (a professional tech educator and system architect). 
        Based on the following transcript, write a highly engaging Title and Description for a full-length YouTube video.
        
        Return ONLY a JSON object with two keys:
        - "title" (A catchy, professional YouTube title)
        - "description" (An engaging 2-3 paragraph YouTube description written in your authentic voice, summarizing the core value of the video)
        
        Transcript:
        {transcript}
        """
        
        response = client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.MINIMAL 
                )
            )
        )
        
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
            
        data = json.loads(raw_text.strip())
        return data
        
    except Exception as e:
        logging.error(f"Long-Form Synthesizer Node failed: {e}")
        return {"title": "Generation Failed", "description": "Could not generate long-form metadata."}

def group_overlapping_clips(clips: list) -> list:
    """Groups clips that overlap, merging their start/end times and concatenating feedback."""
    if not clips:
        return []
        
    sorted_clips = sorted(clips, key=lambda x: x.get('start_time', 0))
    grouped = []
    current_group = [sorted_clips[0]]
    
    for clip in sorted_clips[1:]:
        group_end = max(c.get('end_time', 0) for c in current_group)
        if clip.get('start_time', 0) <= group_end + 5: # 5 seconds buffer
            current_group.append(clip)
        else:
            grouped.append(current_group)
            current_group = [clip]
    if current_group:
        grouped.append(current_group)
        
    merged_clips = []
    for group in grouped:
        main_start = min(c.get('start_time', 0) for c in group)
        main_end = max(c.get('end_time', 0) for c in group)
        
        guardian_clip = next((c for c in group if "GUARDIAN" in c.get('persona_name', '')), None)
        if guardian_clip:
            main_title = guardian_clip.get('title', group[0].get('title', 'Clip'))
        else:
            main_title = group[0].get('title', 'Clip')
            
        feedbacks = []
        for c in group:
            p_name = c.get('persona_name', 'Unknown Persona')
            f_text = c.get('persona_feedback', c.get('description', ''))
            feedbacks.append(f"[{p_name}]\n{f_text}")
            
        merged_clips.append({
            "start_time": main_start,
            "end_time": main_end,
            "title": main_title,
            "persona_feedback": "\n\n".join(feedbacks)
        })
        
    return merged_clips

def send_discord_notification(message: str):
    """Send a notification to Discord."""
    if not DISCORD_WEBHOOK_URL or "YOUR_DISCORD_WEBHOOK_HERE" in DISCORD_WEBHOOK_URL:
        return
    import requests
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": message})
    except Exception as e:
        logging.error(f"Discord notification failed: {e}")

def sync_to_cloud(local_bundle_dir: Path):
    """Move the final bundle to the mounted Google Drive folder."""
    if not GOOGLE_DRIVE_PATH.exists():
        logging.error(f"Cloud mount point {GOOGLE_DRIVE_PATH} not found!")
        return
        
    dest = GOOGLE_DRIVE_PATH / local_bundle_dir.name
    logging.info(f"Syncing bundle to mounted Cloud drive: {dest}")
    try:
        # shutil.copytree can be finicky on mounts, use a safer manual copy if it fails
        shutil.copytree(str(local_bundle_dir), str(dest), dirs_exist_ok=True)
    except Exception as e:
        logging.warning(f"Standard copy failed ({e}), attempting file-by-file copy...")
        dest.mkdir(parents=True, exist_ok=True)
        for item in local_bundle_dir.iterdir():
            s = local_bundle_dir / item.name
            d = dest / item.name
            if s.is_file():
                shutil.copy(str(s), str(d)) # Use copy instead of copy2 to avoid metadata errors on FUSE mounts

# ==========================================
# FFMPEG RENDERING & SLICING
# ==========================================

def burn_subtitles(video_path: Path, ass_path: Path, output_path: Path):
    """Burn Karaoke subtitles onto the video."""
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"ass='{ass_path.name}'",
        "-c:a", "copy", "-c:v", "libx264", "-preset", "fast",
        str(output_path)
    ]
    subprocess.run(cmd, cwd=STAGING_DIR, capture_output=True, check=True)

def get_video_duration(video_path: Path) -> float:
    """Get video duration via ffprobe."""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())

def slice_video(video_path: Path, clip_info: dict, output_dir: Path):
    """
    Slice captioned video with millisecond accuracy.
    Uses re-encoding (libx264/aac) to ensure frame-perfect cuts
    at exactly the requested timestamps.
    """
    try:
        start = float(clip_info["start_time"])
        end = float(clip_info["end_time"])
    except (KeyError, ValueError, TypeError):
        logging.error(f"Invalid timestamps in clip: {clip_info}")
        return
        
    duration = end - start
    if duration <= 0:
        logging.error(f"Invalid duration ({duration}s) for clip: {clip_info}")
        return
        
    # Sanitize title for filesystem
    raw_title = clip_info.get("title", "Clip").replace(" ", "_")
    title = "".join(c for c in raw_title if c.isalnum() or c in ('_', '-'))
    
    output_path = output_dir / f"SHORT_{title}.mp4"
    
    # Jump and Decode: -ss before -i for fast seek + accuracy. 
    # Use -t (duration) because -ss before -i resets input timestamps to 0, making -to act identically to -t.
    cmd = [
        "ffmpeg", "-y", 
        "-ss", str(start), 
        "-i", str(video_path), 
        "-t", str(duration), 
        "-c:v", "libx264", 
        "-c:a", "aac", 
        "-preset", "veryfast", # Added for better speed while maintaining accuracy
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True, check=True)

# ==========================================
# PIPELINE EXECUTION
# ==========================================

def main():
    init_env()
    raw_files = list(RAW_DIR.glob("*.mp4")) + list(RAW_DIR.glob("*.mkv"))
    if not raw_files:
        logging.info("Sleeping... No new recordings found.")
        return
        
    logging.info(f"Nightly batch started: {len(raw_files)} files found.")

    for file in raw_files:
        staging_video = STAGING_DIR / file.name
        shutil.move(str(file), str(staging_video))
        
        try:
            # Step 1: Prep Audio
            audio_path = STAGING_DIR / f"{staging_video.stem}.mp3"
            extract_audio(staging_video, audio_path)
            
            # Step 2: Transcription
            json_path = run_whisper(audio_path)
            
            # Step 3: Subtitles
            ass_path = STAGING_DIR / f"{staging_video.stem}.ass"
            generate_karaoke_ass(json_path, ass_path)
            
            # Step 4: Render
            full_captioned = STAGING_DIR / f"CAPTIONED_{staging_video.name}"
            burn_subtitles(staging_video, ass_path, full_captioned)
            
            # Step 5: AI Director
            with open(json_path, 'r') as f:
                data = json.load(f)
                segments = data.get("segments", [])
                
            # Pass highly detailed WORD-LEVEL timestamps to the LLM
            # This ensures Gemini can choose the exact word to cut on, preventing awkward mid-sentence slices
            word_level_transcript = []
            for s in segments:
                words = s.get("words", [])
                if words:
                    start = words[0]["start"]
                    end = words[-1]["end"]
                    text = s['text'].strip()
                    word_level_transcript.append(f"[{start:.2f} - {end:.2f}] {text}")
            
            transcript_with_times = "\n".join(word_level_transcript)
            
            # Load MKS Personas
            # Exclusively target KERNEL_ prefix to avoid loading test suites (like WT_SUITE)
            persona_files = list(PERSONAS_DIR.glob("KERNEL_*.json"))
            clip_data = []
            
            if not persona_files:
                logging.info("No MKS Personas found in personas/. Using default AI Director.")
                clip_data = get_ai_director_clips(transcript_with_times)
            else:
                logging.info(f"Loaded {len(persona_files)} MKS Personas. Simulating Virtual Focus Group...")
                for p_file in persona_files:
                    with open(p_file, 'r', encoding='utf-8') as pf:
                        try:
                            persona_json = json.load(pf)
                            persona_clips = get_ai_director_clips(transcript_with_times, persona_json)
                            p_name = persona_json.get('kernel_id', 'Unknown Persona')
                            for c in persona_clips:
                                c['persona_name'] = p_name
                            
                            clip_data.extend(persona_clips)
                        except Exception as e:
                            logging.error(f"Failed to parse persona {p_file.name}: {e}")
            
            if not clip_data:
                logging.warning("No clips identified. Skipping rendering.")
                continue
                
            # Merge overlapping clips from different personas
            clip_data = group_overlapping_clips(clip_data)
            
            # Synthesize final unified caption and infographic prompt
            for c in clip_data:
                feedback = c.get("persona_feedback", c.get("description", ""))
                logging.info(f"Synthesizing unified authentic voice for clip: {c.get('title')}")
                synth_data = synthesize_social_post(c.get("title", "Clip"), feedback)
                c["social_post"] = synth_data.get("social_post", feedback)
                c["video_caption"] = synth_data.get("video_caption", feedback)
                c["infographic_prompt"] = synth_data.get("infographic_prompt", "Generation failed.")
            
            # Step 6: Slicing & Bundling
            raw_title = clip_data[0].get('title', 'Clip') if clip_data else 'Clip'
            safe_title = "".join(c for c in raw_title.replace(" ", "_").replace(":", "_") if c.isalnum() or c in ('_', '-'))
            bundle_dir = FINAL_DIR / f"{datetime.now().strftime('%Y-%m-%d')}_{safe_title}"
            bundle_dir.mkdir(parents=True, exist_ok=True)
            
            for cd in clip_data: 
                slice_video(full_captioned, cd, bundle_dir)
            
            # Step 7: Final Move & Descriptions
            shutil.move(str(staging_video), bundle_dir / f"RAW_{staging_video.name}")
            shutil.move(str(audio_path), bundle_dir / audio_path.name)
            shutil.move(str(json_path), bundle_dir / json_path.name)
            shutil.move(str(ass_path), bundle_dir / ass_path.name)
            shutil.move(str(full_captioned), bundle_dir / full_captioned.name)
            
            # Synthesize long-form metadata for the original video
            long_form = synthesize_long_form_metadata(transcript_with_times)
            
            with open(bundle_dir / "social_descriptions.txt", "w") as f:
                f.write("==========================================\n")
                f.write("🎥 FULL VIDEO METADATA\n")
                f.write("==========================================\n\n")
                f.write(f"TITLE: {long_form.get('title', 'N/A')}\n\n")
                f.write(f"DESCRIPTION:\n{long_form.get('description', 'N/A')}\n\n")
                f.write("==========================================\n")
                f.write("✂️ AI DIRECTOR SHORT CLIPS\n")
                f.write("==========================================\n\n")
                
                for cd in clip_data:
                    st = cd.get('start_time', 0.0)
                    et = cd.get('end_time', 0.0)
                    f.write(f"--- {cd.get('title', 'Clip')} [In: {st:.2f} | Out: {et:.2f}] ---\n")
                    f.write(f"PERSONA FEEDBACK:\n{cd.get('persona_feedback', 'No feedback provided.')}\n\n")
                    f.write(f"VIDEO CAPTION (LUKE'S VOICE):\n{cd.get('video_caption', cd.get('description', ''))}\n\n")
            
            with open(bundle_dir / "text_posts_with_infographics.txt", "w") as f:
                f.write("==========================================\n")
                f.write("🖼️ STANDALONE TEXT POSTS & INFOGRAPHICS\n")
                f.write("==========================================\n\n")
                
                for cd in clip_data:
                    f.write(f"--- {cd.get('title', 'Clip')} ---\n")
                    f.write(f"TEXT POST (LUKE'S VOICE):\n{cd.get('social_post', cd.get('description', ''))}\n\n")
                    f.write(f"INFOGRAPHIC PROMPT:\n{cd.get('infographic_prompt', '')}\n\n")
            
            # Step 8: Cloud Sync
            sync_to_cloud(bundle_dir)
            
            logging.info(f"SUCCESS: {file.name}")
            send_discord_notification(f"✅ AI Bestie Pipeline: `{file.name}` processed. {len(clip_data)} shorts created.")
            
        except Exception as e:
            logging.error(f"FAILED {file.name}: {e}")
            send_discord_notification(f"❌ AI Bestie Error: Failed to process `{file.name}`.")

if __name__ == "__main__":
    main()
