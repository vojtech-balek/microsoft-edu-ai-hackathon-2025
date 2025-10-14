# processing.py - Modular file processing service
import os
import time
import cv2
import ffmpeg
from PIL import Image
import base64
import io
import numpy as np
from string import Template
from .openai_service import extract_image_features_with_llm, extract_text_features_with_llm
import random
import pandas as pd
import json
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import PyPDF2
from .speech_service import transcribe_audio_file

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
def process_text_files(file_paths, output_formats, description, target=None):
    t0 = time.time()
    # Step 1: Extract text layer from each PDF file
    extracted_texts = {}
    for file_path in file_paths:
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ''
                for page in reader.pages:
                    text += page.extract_text() or ''
            filename = os.path.basename(file_path)
            extracted_texts[filename] = text
        except Exception as e:
            extracted_texts[os.path.basename(file_path)] = f'<error extracting text: {e}>'
    # Step 2: Select representative texts
    texts_list = list(extracted_texts.values())
    rep_texts = select_representative_images(texts_list, sample_size=1)  # reuse sampling logic
    # Step 3: Build prompt for feature discovery
    dataset_name = "Text Dataset"
    target = target or "<target>"
    examples_str = '\n---\n'.join(rep_texts)
    prompt = prompt_template.substitute(name=dataset_name, description=description or "", target=target, examples=examples_str)
    # Step 4: Feature extraction using LLM (timed)
    feature_spec = extract_text_features_with_llm(rep_texts, prompt=prompt)
    feature_prompt_time = time.time() - t0
    print(f"{feature_prompt_time = }")
    feature_prompt = str(feature_spec[0])
    # Step 5: Feature generation for all texts (parallel, timed)
    def extract_single(text):
        return extract_text_features_with_llm([text], prompt=feature_prompt, feature_gen=True)
    t1 = time.time()
    all_features = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(extract_single, text) for text in texts_list]
        for future in as_completed(futures):
            all_features.append(future.result())
    all_features = [future.result() for future in futures]
    feature_value_time = time.time() - t1
    print(f"{feature_value_time = }")
    # Step 6: Output tabular dataset
    tabular_output = {}
    for filename, features in zip(extracted_texts.keys(), all_features):
        tabular_output[filename] = features

    # Step 7: Postprocess according to output_formats
    output_data = {}
    if not output_formats:
        output_formats = ['json']
    for fmt in output_formats:
        fmt = fmt.lower()
        df = pd.DataFrame([v[0] for v in tabular_output.values()], index=tabular_output.keys())
        if fmt == 'json':
            output_data['json'] = json.dumps(tabular_output, ensure_ascii=False, indent=2)
        elif fmt == 'csv':
            output_data['csv'] = df.to_csv()
            df.to_csv('test.csv')
        elif fmt == 'xlsx':
            xlsx_buffer = BytesIO()
            df.to_excel(xlsx_buffer)
            xlsx_buffer.seek(0)
            output_data['xlsx'] = xlsx_buffer.read()
        elif fmt == 'xml':
            try:
                output_data['xml'] = df.to_xml(root_name='dataset')
            except Exception:
                output_data['xml'] = '<error>XML export failed</error>'
        else:
            output_data[fmt] = f'<error>Unsupported format: {fmt}</error>'
    return {
        'status': 'processed',
        'type': 'text',
        'files': file_paths,
        'output_formats': output_formats,
        'description': description,
        'tabular_output': tabular_output,
        'outputs': output_data,
        'feature_specification': feature_prompt
    }


def build_image_prompt(name, description, rep_images):
    # Use base64 strings directly for representative images
    examples_str = '\n'.join(rep_images)
    return image_prompt_template.substitute(name=name, description=description)

# Main image processing pipeline

def resize_image(img, size=(500, 500)):
    """Resize a PIL image to the given size."""
    return img.resize(size)

def encode_image_to_base64(img, format="JPEG"):
    """Encode a PIL image to base64 string."""
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode()

def create_dataframe_from_tabular(tabular_output):
    """Create a pandas DataFrame from tabular_output, handling lists of dicts."""
    rows = []
    for v in tabular_output.values():
        if isinstance(v, list) and len(v) > 0:
            rows.append(v[0])
        else:
            rows.append(v)
    return pd.DataFrame(rows, index=tabular_output.keys())

def process_image_files(file_paths, output_formats, description=None):
    """Process image files: resize, encode, extract features, and format output."""
    # Step 1: Load and preprocess images
    prompt_start_time = time.time()
    image_base64_list = []
    for path in file_paths:
        img = Image.open(path).convert('RGB')
        img = resize_image(img)
        img_b64 = encode_image_to_base64(img)
        image_base64_list.append(img_b64)
    # Step 2: Select representative images
    rep_images = select_representative_images(image_base64_list, sample_size=20)
    # Step 3: Build prompt for feature discovery
    dataset_name = "Image Dataset"
    prompt = build_image_prompt(dataset_name, description, rep_images)
    prompt_elapsed = time.time() - prompt_start_time
    print(f"Prompt creation time: {prompt_elapsed:.2f} seconds")
    # Step 4: Feature extraction using multimodal LLM
    feature_spec_start = time.time()
    feature_spec = extract_image_features_with_llm(rep_images, prompt=prompt)
    feature_prompt_time = time.time() - feature_spec_start
    print(f"Feature prompt generation time: {feature_prompt_time:.2f} seconds")
    feature_prompt = str(feature_spec[0])
    # Step 5: Feature generation for all images (parallel)
    def extract_single(img_b64):
        return extract_image_features_with_llm([img_b64], prompt=feature_prompt)
    feature_value_start = time.time()
    all_features = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(extract_single, img_b64) for img_b64 in image_base64_list]
        for future in as_completed(futures):
            all_features.append(future.result())
    all_features = [future.result() for future in futures]
    feature_value_time = time.time() - feature_value_start
    print(f"Feature value generation time: {feature_value_time:.2f} seconds")
    # Step 6: Output tabular dataset
    tabular_output = {os.path.basename(fp): features for fp, features in zip(file_paths, all_features)}
    # Step 7: Postprocess according to output_formats
    output_data = {}
    df = create_dataframe_from_tabular(tabular_output)
    if not output_formats:
        output_formats = ['json']
    for fmt in output_formats:
        fmt = fmt.lower()
        if fmt == 'json':
            output_data['json'] = json.dumps(tabular_output, ensure_ascii=False, indent=2)
        elif fmt == 'csv':
            output_data['csv'] = df.to_csv()
        elif fmt == 'xlsx':
            xlsx_buffer = BytesIO()
            df.to_excel(xlsx_buffer)
            xlsx_buffer.seek(0)
            output_data['xlsx'] = xlsx_buffer.read()
        elif fmt == 'xml':
            try:
                output_data['xml'] = df.to_xml(root_name='dataset')
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
        'outputs': output_data,
        'feature_specification': feature_prompt,
        'feature_prompt_time': feature_prompt_time,
        'feature_value_time': feature_value_time
    }

def process_video_files(file_paths, output_formats, description):
    video_path = file_paths[0]
    key_frame_paths = []
    print("processing video")
    try:
        print("extracting key frames")
        key_frame_arrays = extract_key_frames(video_path, frame_limit=5)
        print("key frames done")
        if not key_frame_arrays:
            return {'status': 'error', 'message': 'No key frames found in video.'}
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        for i, frame in enumerate(key_frame_arrays):
            path = os.path.join('uploads', f"{base_name}_keyframe_{i}.jpg")
            cv2.imwrite(path, frame)
            key_frame_paths.append(path)
        image_pipeline_result = process_image_files(key_frame_paths, output_formats, description)

        final_result = {
            'status': 'processed',
            'type': 'video',
            'original_file': video_path,
            'description': description,
            'key_frame_analysis': image_pipeline_result
        }
        return final_result
    finally:
        for path in key_frame_paths:
            if os.path.exists(path):
                os.remove(path)

def select_representative_images(image_arrays, sample_size=20):
    """
    Select a representative sample of images from the dataset.
    Uses random sampling if the dataset is larger than sample_size.
    Returns a list of numpy arrays.
    """
    if len(image_arrays) <= sample_size:
        return image_arrays
    return random.sample(image_arrays, sample_size)


def extract_key_frames(video_path, frame_limit=8, sharpness_threshold=100.0):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return []

    candidates = []
    frame_skip = 15
    frame_count = 0
    prev_gray_frame = None

    while True:
        is_read, frame = cap.read()
        if not is_read:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness_score = cv2.Laplacian(gray_frame, cv2.CV_64F).var()

        if sharpness_score < sharpness_threshold:
            continue

        change_score = 0
        if prev_gray_frame is not None:
            diff = cv2.absdiff(prev_gray_frame, gray_frame)
            change_score = int(diff.sum())

        candidates.append((change_score, frame))
        prev_gray_frame = gray_frame

    candidates.sort(key=lambda item: item[0], reverse=True)

    top_frames_data = candidates[:frame_limit]

    key_frames = [frame for score, frame in top_frames_data]

    cap.release()
    return key_frames

def resize_with_padding(image, target_size=(1024, 1024), background_color="black"):
    image.thumbnail(target_size)

    new_image = Image.new("RGB", target_size, background_color)

    left = (target_size[0] - image.width) // 2
    top = (target_size[1] - image.height) // 2

    new_image.paste(image, (left, top))

    return new_image

# Universal prompt template for text feature extraction
prompt_template = Template("""
{
    "system_message": "IMPORTANT: Return only a valid JSON object with no explanations, text, or markdown!!! Do not include any commentary or introductory text!!!",
    "input_metadata": {
        "dataset_name": "$name",
        "description": "$description",
        "target": "$target",
        "examples": "$examples"
    },
    "task": {
        "steps": [
            "Analyze the provided metadata and examples to determine the domain and context of the dataset.",
            "Identify the key characteristics of the dataset relevant to predicting the target variable.",
            "Extract at least 20 distinct features from the representative images.",
            "Based on provided examples, try to generalize on the domain",
            "List potential high-level categorical and numerical features based on domain knowledge inferred from the dataset description.",
            "Extract additional potential features from dataset examples using syntactic and semantic patterns, ensuring at least 20 distinct features are generated.",
            "If the text implies certain values that match the target, these values may also be extracted as features. In cases where the target has multiple values, each value can be independently derived from the text as a feature if it is contextually appropriate.",
            "For text-based datasets, identify key phrases, structural components, and linguistic patterns that are relevant.",
            "For numerical datasets, identify aggregation patterns, distributional characteristics, and possible transformations.",
            "Group related features into meaningful categories where applicable.",
            "If a feature has more than 15 unique categories, group less frequent categories into an 'Other' class.",
            "For each identified feature, provide a clear name, description, a complete list of possible values, and a specific LLM extraction query."
        ],
        "constraints": [
            "Ensure features are distinct and non-redundant.",
            "Note that the target variable is not explicitly present in the input text.",
            "Prioritize domain-specific insights over generic ones.",
            "Ensure output is a structured, valid JSON format.",
            "For categorical variables, list possible values with domain justification.",
            "For numeric variables, provide possible transformations (e.g., log, mean differences).",
            "The extraction queries must be specific and detailed to ensure high-quality feature generation.",
            "Tailor extraction queries to the domain context of the dataset.",
            "Generate a diverse set of features to maximize potential predictive power."
        ]
    },
    "output_format": {
        "type": "json",
        "structure": {
            "features": [
                {
                    "feature_name": "<Name of the categorical or numerical feature>",
                    "description": "<Short description of what the feature represents and how it relates to the dataset's context>",
                    "possible_values": ["<Value 1>", "<Value 2>", "...", "<Value n>"],
                    "extraction_query": "Identify the '<feature_name>' based on the provided context. Options: '<Value 1>', '<Value 2>', ..., '<Value n>'."
                }
            ]
        }
    }
    }
""")


# Universal prompt template for image feature discovery
image_prompt_template = Template("""
{
    "system_message": "IMPORTANT: Return only a valid JSON object with no explanations, text, or markdown!!! Do not include any commentary or introductory text!!!",
    "input_metadata": {
        "dataset_name": "$name",
        "description": "$description",
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
            "Ensure output is a structured, valid JSON format.",
            "Tailor extraction queries to the domain context of the dataset.",
            "Generate a diverse set of features to maximize potential predictive power."
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
