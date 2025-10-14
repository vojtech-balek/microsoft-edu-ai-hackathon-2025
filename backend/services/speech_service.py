import os
import requests
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

def transcribe_audio_file(audio_file_path: str, model_choice: Literal["high_quality", "fast"] = "fast") -> str:
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

    if not os.path.exists(audio_file_path):
        return f"File not found: {audio_file_path}"

    lower_case_path = audio_file_path.lower()

    if lower_case_path.endswith(".mp4") or lower_case_path.endswith(".m4a"):
        content_type = "audio/mp4"
    elif lower_case_path.endswith(".mp3") or lower_case_path.endswith(".mpeg"):
        content_type = "audio/mpeg"
    elif lower_case_path.endswith(".wav"):
        content_type = "audio/wav"
    else:
        content_type = "application/octet-stream"

    try:
        url = f"{endpoint}openai/deployments/{deployment}/audio/transcriptions?api-version={api_version}"
        with open(audio_file_path, "rb") as f:
            response = requests.post(
                url,
                headers={"api-key": api_key},
                files={"file": (os.path.basename(audio_file_path), f, content_type)},
                data={"model": "gpt-4o-transcribe", "response_format": "json"},
                timeout=300,
            )

        if response.status_code != 200:
            return f"Transcription failed ({response.status_code}): {response.text}"

        return response.json().get("text", "(no text found)")

    except Exception as e:
        return f"Error: {e}"
