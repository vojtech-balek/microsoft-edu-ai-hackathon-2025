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

from .speech_service import transcribe_video_file


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
    # Step 1: Extract text layer from each PDF file
    extracted_texts = {}
    for file_path in file_paths:
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                #TODO if summary
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
    # Step 4: Feature extraction using LLM
    feature_spec = extract_text_features_with_llm(rep_texts, prompt=prompt)
    print(f"{feature_spec = }")
    feature_prompt = str(feature_spec[0])
    # Step 5: Feature generation for all texts (parallel)
    def extract_single(text):
        return extract_text_features_with_llm([text], prompt=feature_prompt, feature_gen=True)
    all_features = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(extract_single, text) for text in texts_list]
        for future in as_completed(futures):
            all_features.append(future.result())
    all_features = [future.result() for future in futures]
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

    feature_prompt = str(feature_spec[0])

    # Step 5: Feature generation for all images (parallel)
    def extract_single(img_b64):
        return extract_image_features_with_llm([img_b64], prompt=feature_prompt)
    all_features = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        print("inside threadpool")
        future_map = {executor.submit(extract_single, img_b64): img_b64 for img_b64 in image_base64_list}

        results_map = {future_map[future]: future.result() for future in as_completed(future_map)}

        all_features = [results_map[img_b64] for img_b64 in image_base64_list]
    # Step 6: Output tabular dataset
    # Map each set of features to its corresponding image filename
    print("before tabular")
    tabular_output = {}
    for file_path, features in zip(file_paths, all_features):
        filename = os.path.basename(file_path)
        tabular_output[filename] = features
    # Step 7: Postprocess according to output_formats
    output_data = {}
    if not output_formats:
        output_formats = ['json']  # Default to JSON if none specified
    for fmt in output_formats:
        fmt = fmt.lower()
        if fmt == 'json':
            output_data['json'] = json.dumps(tabular_output, ensure_ascii=False, indent=2)
        elif fmt == 'csv':
            df = pd.DataFrame.from_dict(tabular_output, orient='index')
            output_data['csv'] = df.to_csv()
        elif fmt == 'xlsx':
            df = pd.DataFrame.from_dict(tabular_output, orient='index')
            xlsx_buffer = BytesIO()
            df.to_excel(xlsx_buffer)
            xlsx_buffer.seek(0)
            output_data['xlsx'] = xlsx_buffer.read()
        elif fmt == 'xml':
            try:
                df = pd.DataFrame.from_dict(tabular_output, orient='index')
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
        'feature_specification': feature_prompt
    }


def process_video_files(file_paths, output_formats, description):
    consolidated_tabular_output = {}
    feature_spec_from_text = None
    all_transcripts = {}
    all_summaries = {}

    for video_path in file_paths:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        temp_summary_path = None

        try:
            transcript = transcribe_video_file(video_path)
            all_transcripts[os.path.basename(video_path)] = transcript

            key_frame_arrays = extract_key_frames(video_path, frame_limit=5)
            if not key_frame_arrays:
                consolidated_tabular_output[os.path.basename(video_path)] = {"error": "No key frames found."}
                all_summaries[os.path.basename(video_path)] = ""
                continue

            frame_b64_list = []
            for frame in key_frame_arrays:
                if isinstance(frame, np.ndarray):
                    if frame.ndim == 3 and frame.shape[2] == 3:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        frame_rgb = frame
                    pil_image = Image.fromarray(frame_rgb.astype('uint8'), 'RGB')
                else:
                    pil_image = frame

                buffered = io.BytesIO()
                pil_image.save(buffered, format="JPEG")
                frame_b64 = base64.b64encode(buffered.getvalue()).decode()
                frame_b64_list.append(frame_b64)

            summary_prompt = (
                "You are an expert video analyst with deep knowledge in multimodal understanding (audio + visual). "
                "Your task is to produce an exceptionally detailed, comprehensive, and structured summary of the provided file. "
                "Use BOTH the full transcript AND the key visual frames to create a rich, insightful analysis. "
                "Do NOT be brief. Write a long, thorough summary (at least 300–500 words) that could serve as a standalone document "
                "for someone who has not seen the video.\n\n"

                "Structure your response as follows:\n"
                "1. **Context & Purpose**: What is the video about? Who is the audience? What is its goal (educational, promotional, documentary, etc.)?\n"
                "2. **Narrative & Key Topics**: Walk through the main ideas in order. What arguments, explanations, or stories are presented? Include specific examples or quotes if relevant.\n"
                "3. **Visual Analysis**: Describe important visual elements from the frames (e.g., people, settings, on-screen text, diagrams, actions, emotions, scene changes). How do they support or expand the spoken content?\n"
                "4. **Key Takeaways**: What are the 3–5 most important insights, conclusions, or messages?\n"
                "5. **Overall Tone & Style**: Is the video formal, casual, urgent, humorous, technical? How does this affect the message?\n\n"

                "Be precise, analytical, and exhaustive. If the transcript is in Czech, respond in Czech. "
                "If it's in English, respond in English. Match the language of the transcript.\n\n"

                f"Transcript:\n{transcript}"
            )
            video_summary = extract_image_features_with_llm(frame_b64_list, prompt=summary_prompt)
            print(video_summary)
            if not isinstance(video_summary, str):
                video_summary = str(video_summary)

            all_summaries[os.path.basename(video_path)] = video_summary

            os.makedirs('uploads', exist_ok=True)
            temp_summary_path = os.path.join('uploads', f"summary_{base_name}.txt")
            with open(temp_summary_path, 'w', encoding='utf-8') as f:
                f.write(video_summary)

            single_video_result = process_text_files([temp_summary_path], output_formats, description)
            summary_filename = os.path.basename(temp_summary_path)
            if single_video_result.get('tabular_output', {}).get(summary_filename):
                consolidated_tabular_output[os.path.basename(video_path)] = single_video_result['tabular_output'][summary_filename]
            feature_spec_from_text = single_video_result.get('feature_specification')

        finally:
            if temp_summary_path and os.path.exists(temp_summary_path):
                os.remove(temp_summary_path)

    output_data = {}
    if not output_formats:
        output_formats = ['json']

    for fmt in output_formats:
        fmt = fmt.lower()
        if fmt == 'json':
            output_data['json'] = json.dumps(consolidated_tabular_output, ensure_ascii=False, indent=2)
        elif consolidated_tabular_output:
            df = pd.DataFrame.from_dict(consolidated_tabular_output, orient='index')
            if not df.empty and isinstance(df.iloc[0, 0], list) and len(df.iloc[0, 0]) == 1:
                df = df.applymap(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)

            if fmt == 'csv':
                output_data['csv'] = df.to_csv()
            elif fmt == 'xlsx':
                xlsx_buffer = BytesIO()
                df.to_excel(xlsx_buffer, index=True)
                xlsx_buffer.seek(0)
                output_data['xlsx'] = xlsx_buffer.read()
            elif fmt == 'xml':
                try:
                    output_data['xml'] = df.to_xml(root_name='dataset')
                except Exception:
                    output_data['xml'] = '<error>XML export failed</error>'


    return {
        'status': 'processed',
        'type': 'video',
        'original_files': file_paths,
        'output_formats': output_formats,
        'description': description,
        'tabular_output': consolidated_tabular_output,
        'outputs': output_data,
        'feature_specification': feature_spec_from_text,
        'transcripts': all_transcripts,
        'summaries': all_summaries
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


def create_collage(image_arrays, collage_width=1024):
    if not image_arrays:
        return None

    target_height = collage_width // len(image_arrays)
    pil_images = []
    total_width = 0
    for frame in image_arrays:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)

        ratio = target_height / img.height
        new_width = int(img.width * ratio)
        img = img.resize((new_width, target_height), Image.LANCZOS)

        pil_images.append(img)
        total_width += new_width

    collage = Image.new('RGB', (total_width, target_height))

    current_x = 0
    for img in pil_images:
        collage.paste(img, (current_x, 0))
        current_x += img.width

    return collage
