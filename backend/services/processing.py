# processing.py - Modular file processing service
import os
from PIL import Image
import base64
import io
import numpy as np
from string import Template
from .openai_service import extract_image_features_with_llm
import random
import pandas as pd
import json
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_files(file_paths, file_type, output_formats=None, description=None):
    if output_formats is None:
        output_formats = []
    result = None
    if file_type == 'text':
        result = process_text_files(file_paths, output_formats, description)
    elif file_type == 'image':
        result = process_image_files(file_paths, output_formats, description)
    elif file_type == 'video':
        result = process_video_files(file_paths, output_formats, description)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    return result

# Example text file processing (PDF)
def process_text_files(file_paths, output_formats, description):
    # Placeholder: implement PDF parsing, text extraction, etc.
    return {
        'status': 'processed',
        'type': 'text',
        'files': file_paths,
        'output_formats': output_formats,
        'description': description
    }

# Universal prompt template for image feature discovery
image_prompt_template = Template("""
{
    "system_message": "IMPORTANT: Return only a valid JSON object with no explanations, text, or markdown!!! Do not include any commentary or introductory text!!!",
    "input_metadata": {
        "dataset_name": "$name",
        "description": "$description",
        "representative_images": "$examples"
    },
    "task": {
        "steps": [
            "Analyze the provided metadata and representative images to determine the domain and context of the dataset.",
            "Identify key visual characteristics relevant to feature extraction.",
            "List potential high-level categorical and numerical features based on domain knowledge and image content.",
            "Extract at least 10 distinct features from the representative images.",
            "For each identified feature, provide a clear name, description, possible values, and a specific LLM extraction query."
        ],
        "constraints": [
            "Ensure features are distinct and non-redundant.",
            "Prioritize domain-specific insights over generic ones.",
            "Ensure output is a structured, valid JSON format."
        ]
    },
    "output_format": {
        "type": "json",
        "structure": {
            "features": [
                {
                    "feature_name": "<Name>",
                    "description": "<Description>",
                    "possible_values": ["<Value 1>", "<Value 2>", ...],
                    "extraction_query": "<Query>"
                }
            ]
        }
    }
}
""")

def build_image_prompt(name, description, rep_images):
    # Use base64 strings directly for representative images
    examples_str = '\n'.join(rep_images)
    return image_prompt_template.substitute(name=name, description=description, examples=examples_str)

# Main image processing pipeline

def process_image_files(file_paths, output_formats, description=None):
    # Step 1: Load images and store as base64
    image_base64_list = []
    for path in file_paths:
        img = Image.open(path).convert('RGB')
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        image_base64_list.append(img_b64)
    # Step 2: Select representative images (base64)
    rep_images = select_representative_images(image_base64_list, sample_size=20)
    # Step 3: Build prompt for feature discovery
    dataset_name = "Image Dataset"
    prompt = build_image_prompt(dataset_name, description, rep_images)
    # Step 4: Feature extraction using multimodal LLM
    feature_spec = extract_image_features_with_llm(rep_images, prompt=prompt)
    # Ensure feature_spec is a dict and get features for prompt
    if isinstance(feature_spec, dict) and 'features' in feature_spec:
        feature_prompt = json.dumps(feature_spec['features'])
    elif isinstance(feature_spec, list) and len(feature_spec) > 0:
        feature_prompt = str(feature_spec[0])
    else:
        feature_prompt = str(feature_spec)
    # Step 5: Feature generation for all images (parallel)
    def extract_single(img_b64):
        return extract_image_features_with_llm([img_b64], prompt=feature_prompt)
    all_features = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(extract_single, img_b64) for img_b64 in image_base64_list]
        for future in as_completed(futures):
            all_features.append(future.result())
    # Preserve original order
    all_features = [future.result() for future in futures]
    # Step 6: Output tabular dataset
    df = pd.DataFrame(all_features)
    tabular_output = df.to_dict(orient='records')
    # Step 7: Postprocess according to output_formats
    output_data = {}
    if not output_formats:
        output_formats = ['json']  # Default to JSON if none specified
    for fmt in output_formats:
        fmt = fmt.lower()
        if fmt == 'json':
            output_data['json'] = json.dumps(tabular_output, ensure_ascii=False, indent=2)
        elif fmt == 'csv':
            output_data['csv'] = df.to_csv(index=False)
        elif fmt == 'xlsx':
            xlsx_buffer = BytesIO()
            df.to_excel(xlsx_buffer, index=False)
            xlsx_buffer.seek(0)
            output_data['xlsx'] = xlsx_buffer.read()
        elif fmt == 'xml':
            try:
                output_data['xml'] = df.to_xml(root_name='dataset', index=False)
            except Exception:
                output_data['xml'] = '<error>XML export failed</error>'
        else:
            output_data[fmt] = f'<error>Unsupported format: {fmt}</error>'
    return {
        'status': 'processed',
        'type': 'image',
        'original_files': file_paths,
        'output_formats': output_formats,
        'description': description,
        'tabular_output': tabular_output,
        'outputs': output_data
    }

# Example video file processing
def process_video_files(file_paths, output_formats, description):
    # Placeholder: implement video analysis, transcoding, etc.
    return {
        'status': 'processed',
        'type': 'video',
        'files': file_paths,
        'output_formats': output_formats,
        'description': description
    }

def select_representative_images(image_arrays, sample_size=5):
    """
    Select a representative sample of images from the dataset.
    Uses random sampling if the dataset is larger than sample_size.
    Returns a list of numpy arrays.
    """
    if len(image_arrays) <= sample_size:
        return image_arrays
    return random.sample(image_arrays, sample_size)
