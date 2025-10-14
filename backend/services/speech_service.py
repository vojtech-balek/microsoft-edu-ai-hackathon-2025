import os
import requests
import ffmpeg
import time
from typing import Literal
from dotenv import load_dotenv

load_dotenv()


def transcribe_video_file(file_path: str, model_choice: Literal["high_quality", "fast"] = "fast") -> str:
    endpoint = os.getenv("SPEECH_AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("SPEECH_AZURE_OPENAI_API_KEY")
    api_version = os.getenv("SPEECH_AZURE_OPENAI_API_VERSION")

    if not endpoint or not api_key:
        return "Speech service is not configured."

    deployment = (
        os.getenv("SPEECH_GPT4O_TRANSCRIBE_DEPLOYMENT_NAME")
        if model_choice == "high_quality"
        else os.getenv("SPEECH_GPT4O_MINI_TRANSCRIBE_DEPLOYMENT_NAME")
    )

    if not deployment:
        return f"Deployment for '{model_choice}' not found."

    if not os.path.exists(file_path):
        return f"File not found: {file_path}"

    video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
    path_to_process = file_path
    temp_audio_path = None

    try:
        _, extension = os.path.splitext(file_path.lower())
        if extension in video_extensions:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            temp_audio_path = os.path.join('uploads', f"temp_audio_{base_name}_{int(time.time())}.wav")

            try:
                ffmpeg.input(file_path).output(
                    temp_audio_path, acodec='pcm_s16le', ac=1, ar='16k'
                ).run(quiet=True, overwrite_output=True)
            except ffmpeg.Error as e:
                error_details = e.stderr.decode('utf8') if e.stderr else 'Unknown FFmpeg error'
                return f"Failed to extract audio from video: {error_details}"

            path_to_process = temp_audio_path

        lower_case_path = path_to_process.lower()

        if lower_case_path.endswith(".mp4") or lower_case_path.endswith(".m4a"):
            content_type = "audio/mp4"
        elif lower_case_path.endswith(".mp3") or lower_case_path.endswith(".mpeg"):
            content_type = "audio/mpeg"
        elif lower_case_path.endswith(".wav"):
            content_type = "audio/wav"
        else:
            content_type = "application/octet-stream"

        url = f"{endpoint}openai/deployments/{deployment}/audio/transcriptions?api-version={api_version}"
        with open(path_to_process, "rb") as f:
            response = requests.post(
                url,
                headers={"api-key": api_key},
                files={"file": (os.path.basename(path_to_process), f, content_type)},
                data={"model": "gpt-4o-transcribe", "response_format": "json"},
                timeout=300,
            )

        if response.status_code != 200:
            return f"Transcription failed ({response.status_code}): {response.text}"

        return response.json().get("text", "(no text found)")

    except Exception as e:
        return f"An error occurred during the API call: {e}"
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)