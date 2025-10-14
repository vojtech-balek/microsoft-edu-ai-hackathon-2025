import os
import openai
import numpy as np
from PIL import Image
import base64
import io
from dotenv import load_dotenv
import json
import time

load_dotenv()

client = openai.AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

def image_to_base64(img_arr):
    img = Image.fromarray(img_arr.astype(np.uint8))
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def extract_image_features_with_llm(image_base64_list, prompt=None, deployment_name=None) -> list:
    features_list = []
    for img_b64 in image_base64_list:
        if prompt is None:
            prompt_text = "Extract meaningful features from this image for tabular dataset construction."
        else:
            prompt_text = prompt
        if deployment_name is None:
            deployment_name = os.getenv("AZURE_OPENAI_GPT41_DEPLOYMENT_NAME")
        max_retries = 5
        backoff = 2
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=deployment_name,
                    messages=[
                        {"role": "system", "content": "You are a feature extraction assistant for images."},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                            ]
                        }
                    ],
                    max_tokens=512,
                    temperature=0.0
                )
                content = response.choices[0].message.content
                try:
                    features = json.loads(content)
                except Exception:
                    features = {"features": content}
                features_list.append(features)
                break
            except openai.RateLimitError:
                if attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    features_list.append({"error": "Rate limit exceeded. Please try again later."})
            except Exception as e:
                features_list.append({"error": str(e)})
                break
    return features_list

def extract_text_features_with_llm(text_list, prompt=None, deployment_name=None, feature_gen = False) -> list:
    features_list = []
    for text in text_list:
        if prompt is None:
            prompt_text = "Extract meaningful features from this text for tabular dataset construction."
        else:
            prompt_text = prompt
        if deployment_name is None:
            deployment_name = os.getenv("AZURE_OPENAI_GPT41_DEPLOYMENT_NAME")
        max_retries = 5
        backoff = 2
        if feature_gen:
            prompt_text = """You are a feature extraction assistant for text documents. 
                        You provide output in a structured JSON format and do NOT provide any explanations.
                        {
                          "<feature1_name>": "<value1>",
                          "<feature2_name>": "<value2>",
                          "<feature3_name>": "<value3>",
                          "<feature4_name>": "<value4>",
                          "<feature5_name>": "<value5>"
                        }""" + prompt
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=deployment_name,
                    messages=[
                        {"role": "system", "content": prompt_text
                         },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text}
                            ]
                        }
                    ],
                    max_tokens=512,
                    temperature=0.0
                )
                content = response.choices[0].message.content
                try:
                    features = json.loads(content)
                except Exception:
                    features = {"features": content}
                features_list.append(features)
                break
            except openai.RateLimitError:
                if attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    features_list.append({"error": "Rate limit exceeded. Please try again later."})
            except Exception as e:
                features_list.append({"error": str(e)})
                break
    return features_list
