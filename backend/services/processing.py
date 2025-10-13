# processing.py - Modular file processing service
import os
from PIL import Image
import numpy as np
from string import Template
from services.openai_service import extract_image_features_with_llm
import random
import pandas as pd
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

# Image resizing function
def resize_images(image_paths, target_size=(224, 224), output_dir=None, save_to_disk=True):
    resized_outputs = []
    if output_dir is None:
        output_dir = os.path.dirname(image_paths[0]) if image_paths else '.'
    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                # Preserve aspect ratio and pad
                img.thumbnail(target_size, Image.LANCZOS)
                new_img = Image.new('RGB', target_size, (0, 0, 0))  # black padding
                left = (target_size[0] - img.width) // 2
                top = (target_size[1] - img.height) // 2
                new_img.paste(img, (left, top))
                if save_to_disk:
                    base, ext = os.path.splitext(os.path.basename(img_path))
                    resized_name = f"{base}_resized{ext}"
                    resized_path = os.path.join(output_dir, resized_name)
                    new_img.save(resized_path)
                    resized_outputs.append(resized_path)
                else:
                    arr = np.array(new_img)
                    resized_outputs.append(arr)
        except Exception as e:
            print(f"Error resizing {img_path}: {e}")
    return resized_outputs

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
            "Extract at least 20 distinct features from the representative images.",
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
    # Convert representative images to base64 strings for prompt
    examples = [str(img.tolist()) for img in rep_images]  # For LLM, you may use base64 or summary
    examples_str = '\n'.join(examples)
    return image_prompt_template.substitute(name=name, description=description, examples=examples_str)

# Main image processing pipeline

def process_image_files(file_paths, output_formats, description, target_size=(224, 224), save_to_disk=False, dataset_name="Image Dataset"):
    # Step 1: Preprocess images
    image_arrays = resize_images(file_paths, target_size, save_to_disk=save_to_disk)
    # Step 2: Select representative images
    rep_images = select_representative_images(image_arrays, sample_size=20)
    # Step 3: Build prompt for feature discovery
    prompt = build_image_prompt(dataset_name, description, rep_images)
    # Step 4: Feature extraction using multimodal LLM
    feature_spec = extract_image_features_with_llm([rep_images], prompt=prompt)
    # Step 5: Feature generation for all images
    all_features = []
    for img_arr in image_arrays:
        features = extract_image_features_with_llm([img_arr], prompt=feature_spec.get('features', []))
        all_features.append(features)
    # Step 6: Tabular output
    df = pd.DataFrame(all_features)
    tabular_output = df.to_dict(orient='records')
    return {
        'status': 'processed',
        'type': 'image',
        'original_files': file_paths,
        'output_formats': output_formats,
        'description': description,
        'resolution': target_size,
        'tabular_output': tabular_output
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

def select_representative_images(image_arrays, sample_size=20):
    """
    Select a representative sample of images from the dataset.
    Uses random sampling if the dataset is larger than sample_size.
    Returns a list of numpy arrays.
    """
    if len(image_arrays) <= sample_size:
        return image_arrays
    return random.sample(image_arrays, sample_size)
