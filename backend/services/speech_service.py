import os
from typing import Literal

import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

load_dotenv()

SPEECH_API_KEY = os.getenv("SPEECH_AZURE_OPENAI_API_KEY")
SPEECH_REGION = os.getenv("SPEECH_AZURE_REGION")
HIGH_QUALITY_MODEL_NAME = os.getenv("MEGATHINK_TRANSCRIBE_HIGH_QUALITY")
FAST_MODEL_NAME = os.getenv("MEGATHINK_TRANSCRIBE_FAST")

ENDPOINT = f"wss://{SPEECH_REGION}.stt.speech.microsoft.com/speech/universal/v2" if SPEECH_REGION else None


def transcribe_audio_file(
    audio_file_path: str, model_choice: Literal["high_quality", "fast"] = "fast"
) -> str:
    if not audio_file_path or not os.path.exists(audio_file_path):
        return "Audio file not found."

    if not SPEECH_API_KEY or not SPEECH_REGION:
        return "Missing Azure Speech configuration: SPEECH_AZURE_OPENAI_API_KEY or SPEECH_AZURE_REGION."

    deployment_name = FAST_MODEL_NAME if model_choice == "fast" else HIGH_QUALITY_MODEL_NAME
    if not deployment_name:
        return "Missing deployment name for the selected model."

    try:
        speech_config = speechsdk.SpeechConfig(subscription=SPEECH_API_KEY, endpoint=ENDPOINT)
        speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_EndpointId, deployment_name)

        audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        result = recognizer.recognize_once()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text or ""
        if result.reason == speechsdk.ResultReason.NoMatch:
            return "No speech could be recognized."
        if result.reason == speechsdk.ResultReason.Canceled:
            details = result.cancellation_details
            return f"Transcription canceled: {details.reason}. Error: {details.error_details}"

        return "Unknown transcription result."
    except Exception as e:
        return f"An exception occurred: {e}"