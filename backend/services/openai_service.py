import os
import openai
import numpy as np
from PIL import Image
import base64
import io
from dotenv import load_dotenv

load_dotenv()

# Azure OpenAI config from .env
OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# Deployment names for different models
GPT41_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_GPT41_DEPLOYMENT_NAME")
GPT41_MINI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_GPT41_MINI_DEPLOYMENT_NAME")
O3_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_O3_DEPLOYMENT_NAME")

# Megathink model selection
MEGATHINK_QUICK_MODEL = os.getenv("MEGATHINK_QUICK_MODEL", GPT41_MINI_DEPLOYMENT_NAME)
MEGATHINK_COMPLEX_MODEL = os.getenv("MEGATHINK_COMPLEX_MODEL", GPT41_DEPLOYMENT_NAME)
MEGATHINK_REASONING_MODEL = os.getenv("MEGATHINK_REASONING_MODEL", O3_DEPLOYMENT_NAME)
MEGATHINK_DEFAULT_MODEL = os.getenv("MEGATHINK_DEFAULT_MODEL", GPT41_DEPLOYMENT_NAME)

openai.api_type = "azure"
openai.api_key = OPENAI_API_KEY
openai.api_base = OPENAI_API_BASE
openai.api_version = OPENAI_API_VERSION

def image_to_base64(img_arr):
    img = Image.fromarray(img_arr.astype(np.uint8))
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def extract_image_features_with_llm(image_arrays, prompt=None, deployment_name=None):
    features_list = []
    for img_arr in image_arrays:
        img_b64 = image_to_base64(img_arr)
        if prompt is None:
            prompt_text = "Extract meaningful features from this image for tabular dataset construction."
        else:
            prompt_text = prompt
        if deployment_name is None:
            deployment_name = MEGATHINK_DEFAULT_MODEL
        response = openai.ChatCompletion.create(
            engine=deployment_name,
            messages=[
                {"role": "system", "content": "You are a feature extraction assistant for images."},
                {"role": "user", "content": prompt_text},
                {"role": "user", "content": {"image": img_b64}}
            ],
            max_tokens=512
        )
        content = response.choices[0].message["content"]
        try:
            import json
            features = json.loads(content)
        except Exception:
            features = {"features": content}
        features_list.append(features)
    if len(features_list) == 1:
        return features_list[0]
    return features_list

