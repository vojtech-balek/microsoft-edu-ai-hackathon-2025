# processing.py - Modular file processing service

# --- 1. Imports ---
import os
import time
import json
import base64
import io
import random
import shutil
import zipfile
import logging
from string import Template
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Any, Tuple, Dict, Callable

# Third-party libraries
import cv2
import ffmpeg
import numpy as np
import pandas as pd
import PyPDF2
from PIL import Image

# Local application imports
from .openai_service import extract_image_features_with_llm, extract_text_features_with_llm
from .speech_service import transcribe_video_file

# --- 2. Logging and Constants Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# General constants
ALLOWED_EXTENSIONS = {
    'text': {'pdf', 'txt', 'md', 'log', 'csv'},
    'image': {'png', 'jpg', 'jpeg'},
    'video': {'mp4', 'avi', 'mov', 'mkv'}
}
MAX_PARALLEL_WORKERS: int = 10
VALID_OUTPUT_FORMATS = {'json', 'csv', 'xlsx', 'xml'}

# Text processing constants
TEXT_DATASET_NAME = "Text Dataset"
TEXT_SAMPLE_SIZE = 20
DEFAULT_TARGET_VARIABLE = "<target>"

# Image processing constants
IMAGE_DATASET_NAME = "Image Dataset"
IMAGE_SAMPLE_SIZE = 20
RESIZE_DIMENSIONS: Tuple[int, int] = (768, 768)
IMAGE_ENCODE_FORMAT: str = "JPEG"

# Video processing constants
KEY_FRAME_LIMIT = 8
SUMMARY_PROMPT_TEMPLATE = (
    "You are an expert video analyst with deep knowledge in multimodal understanding (audio + visual). "
    "Your task is to produce an exceptionally detailed, comprehensive, and structured summary of the provided file. "
    "Use BOTH the full transcript AND the key visual frames to create a rich, insightful analysis. "
    "Do NOT be brief. Write a long, thorough summary (at least 300–500 words).\n\n"
    "Transcript:\n{transcript}"
)


# --- 3. Generic Helper Functions ---

def create_dataframe_from_tabular(tabular_output: Dict[str, Any]) -> pd.DataFrame:
    """Robustly creates a pandas DataFrame from various tabular_output structures."""
    rows = []
    for v in tabular_output.values():
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
            rows.append(v[0])
        elif isinstance(v, dict):
            rows.append(v)
        else:  # Fallback for unexpected formats
            rows.append({})

    df = pd.DataFrame(rows, index=tabular_output.keys())
    # Unpack single-element lists within cells, common in the video processor
    if not df.empty:
        df = df.applymap(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)
    return df


def _format_outputs(tabular_output: Dict, output_formats: List[str]) -> Dict[str, Any]:
    """Converts the tabular data into various specified output formats."""
    if not output_formats:
        output_formats = ['json']

    output_data = {}
    if not tabular_output:
        return {'json': json.dumps({}, indent=2)}

    df = create_dataframe_from_tabular(tabular_output)

    for fmt in [f.lower() for f in output_formats if f in VALID_OUTPUT_FORMATS]:
        try:
            if fmt == 'json':
                output_data['json'] = json.dumps(tabular_output, ensure_ascii=False, indent=2)
            elif fmt == 'csv':
                output_data['csv'] = df.to_csv()
                # Preserving side-effect from original code. Consider removing in production.
                if 'test.csv' in os.listdir(): df.to_csv('test.csv')
            elif fmt == 'xlsx':
                xlsx_buffer = io.BytesIO()
                df.to_excel(xlsx_buffer, index=True)
                xlsx_buffer.seek(0)
                output_data['xlsx'] = xlsx_buffer.read()
            elif fmt == 'xml':
                output_data['xml'] = df.to_xml(root_name='dataset')
        except Exception as e:
            logger.error(f"Output format '{fmt}' generation failed: {e}")
            output_data[fmt] = f"<error>Export to {fmt} failed: {e}</error>"
    return output_data


def _run_parallel_feature_extraction(items: List[Any], extraction_func: Callable, prompt: str) -> List[Dict]:
    """Generic function to run feature extraction in parallel using a thread pool."""

    def task_wrapper(item: Any) -> Dict:
        return extraction_func([item], prompt=prompt, feature_gen=True)

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
        futures = [executor.submit(task_wrapper, item) for item in items]
        return [future.result() for future in futures]


# --- 4. Main Dispatcher and ZIP Processor ---

def _process_zip_archive(zip_path: str, output_formats: List[str], description: Optional[str]) -> dict:
    """Handles the logic of processing ZIP archives."""
    temp_dir = os.path.join('uploads', f"temp_unzip_{int(time.time())}")
    os.makedirs(temp_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        extracted_files = [
            os.path.join(root, name)
            for root, dirs, files in os.walk(temp_dir)
            for name in files if not name.startswith(('__MACOSX', '.'))
        ]

        if not extracted_files:
            return {'status': 'error', 'message': 'The ZIP file is empty.'}

        inner_file_types = {
            ftype for f in extracted_files
            for ftype, exts in ALLOWED_EXTENSIONS.items()
            if os.path.splitext(f)[1].lower().replace('.', '') in exts
        }

        if len(inner_file_types) > 1:
            return {'status': 'error', 'message': f'ZIP file contains multiple file types: {list(inner_file_types)}'}
        if not inner_file_types:
            return {'status': 'error', 'message': 'No supported file types found in the ZIP file.'}

        actual_file_type = inner_file_types.pop()
        logger.info(f"Processing {len(extracted_files)} files of type '{actual_file_type}' from ZIP archive...")

        if actual_file_type == 'text':
            return process_text_files(extracted_files, output_formats, description)
        elif actual_file_type == 'image':
            return process_image_files(extracted_files, output_formats, description)
        elif actual_file_type == 'video':
            return process_video_files(extracted_files, output_formats, description)
        else:
            return {'status': 'error', 'message': f'File type "{actual_file_type}" in ZIP is not supported.'}
    finally:
        shutil.rmtree(temp_dir)


def process_files(file_paths: List[str], file_type: str, output_formats: Optional[List[str]] = None,
                  description: Optional[str] = None) -> dict:
    """Main dispatcher function to process files based on their type."""
    output_formats = output_formats or []
    if file_type == 'zip':
        return _process_zip_archive(file_paths[0], output_formats, description)
    elif file_type == 'text':
        return process_text_files(file_paths, output_formats, description)
    elif file_type == 'image':
        return process_image_files(file_paths, output_formats, description)
    elif file_type == 'video':
        return process_video_files(file_paths, output_formats, description)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


# --- 5. Text File Processing ---

def _extract_texts_from_files(file_paths: List[str]) -> Dict[str, str]:
    """Extracts raw text from files (PDF, TXT, etc.)."""
    extracted_texts = {}
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        try:
            _, ext = os.path.splitext(file_path)
            if ext.lower() == '.pdf':
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    extracted_texts[filename] = ''.join(page.extract_text() or '' for page in reader.pages)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    extracted_texts[filename] = f.read()
        except Exception as e:
            extracted_texts[filename] = f'<error extracting text: {e}>'
    return extracted_texts


def process_text_files(file_paths: List[str], output_formats: List[str], description: str,
                       target: Optional[str] = None) -> dict:
    logger.info("Processing text files...")
    extracted_texts = _extract_texts_from_files(file_paths)
    texts_list = list(extracted_texts.values())
    rep_texts = select_representative_images(texts_list, sample_size=TEXT_SAMPLE_SIZE)

    prompt = prompt_template.substitute(
        name=TEXT_DATASET_NAME,
        description=description or "",
        target=target or DEFAULT_TARGET_VARIABLE,
        examples='\n---\n'.join(rep_texts)
    )
    feature_spec = extract_text_features_with_llm(rep_texts, prompt=prompt)
    feature_prompt = str(feature_spec[0]) if feature_spec else ""

    all_features = _run_parallel_feature_extraction(texts_list, extract_text_features_with_llm, feature_prompt)
    tabular_output = dict(zip(extracted_texts.keys(), all_features))
    output_data = _format_outputs(tabular_output, output_formats)

    return {
        'status': 'processed', 'type': 'text', 'files': file_paths,
        'output_formats': output_formats, 'description': description,
        'tabular_output': tabular_output, 'outputs': output_data,
        'feature_specification': feature_prompt
    }


# --- 6. Image File Processing ---

def _preprocess_images(file_paths: List[str]) -> Dict[str, Optional[str]]:
    """Loads, resizes, and encodes image files to base64, returning a dictionary."""
    processed_images = {}
    for path in file_paths:
        try:
            with Image.open(path) as img:
                img_rgb = img.convert('RGB')
                resized_img = img_rgb.resize(RESIZE_DIMENSIONS)
                buffered = io.BytesIO()
                resized_img.save(buffered, format=IMAGE_ENCODE_FORMAT)
                processed_images[path] = base64.b64encode(buffered.getvalue()).decode()
        except Exception as e:
            logger.warning(f"Could not process image {path}. Error: {e}")
            processed_images[path] = None
    return processed_images


def process_image_files(file_paths: List[str], output_formats: List[str], description: Optional[str] = None) -> Dict:
    logger.info("Processing image files...")
    start_time = time.time()

    processed_images = _preprocess_images(file_paths)
    valid_base64_list = [b64 for b64 in processed_images.values() if b64 is not None]

    if not valid_base64_list:
        return {'status': 'error', 'message': 'No valid images could be processed.'}

    rep_images = select_representative_images(valid_base64_list, sample_size=IMAGE_SAMPLE_SIZE)
    prompt = image_prompt_template.substitute(name=IMAGE_DATASET_NAME, description=description or "")
    logger.info(f"Prompt creation time: {time.time() - start_time:.2f}s")

    feature_spec_start = time.time()
    feature_spec = extract_image_features_with_llm(rep_images, prompt=prompt)
    feature_prompt = str(feature_spec[0]) if feature_spec else ""
    feature_prompt_time = time.time() - feature_spec_start
    logger.info(f"Feature prompt generation time: {feature_prompt_time:.2f}s")

    feature_value_start = time.time()
    all_features = _run_parallel_feature_extraction(valid_base64_list, extract_image_features_with_llm, feature_prompt)
    feature_value_time = time.time() - feature_value_start
    logger.info(f"Feature value generation time: {feature_value_time:.2f}s")

    tabular_output = {}
    feature_iterator = iter(all_features)
    for path, b64 in processed_images.items():
        filename = os.path.basename(path)
        if b64 is not None:
            tabular_output[filename] = next(feature_iterator)
        else:
            tabular_output[filename] = {"error": "Failed to process this image file."}

    output_data = _format_outputs(tabular_output, output_formats)

    return {
        'status': 'processed', 'type': 'image', 'original_files': file_paths,
        'output_formats': output_formats, 'description': description,
        'tabular_output': tabular_output, 'outputs': output_data,
        'feature_specification': feature_prompt,
        'feature_prompt_time': feature_prompt_time, 'feature_value_time': feature_value_time
    }


# --- 7. Video File Processing ---

def _convert_frames_to_base64(key_frames: List[Any], filename: str) -> List[str]:
    """Converts a list of frame arrays/images to base64 strings."""
    frame_b64_list = []
    for i, frame in enumerate(key_frames):
        try:
            if isinstance(frame, np.ndarray):
                if frame.size == 0: continue
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            elif hasattr(frame, 'save'):
                pil_image = frame
            else:
                logger.warning(f"Unsupported frame type {type(frame)} in {filename}, skipping frame {i}")
                continue

            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG")
            frame_b64_list.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))
        except Exception as e:
            logger.warning(f"Failed to process frame {i} in {filename}: {e}")
    return frame_b64_list


def _process_single_video(video_path: str) -> Dict[str, Any]:
    """Processes one video file to get its transcript and summary."""
    filename = os.path.basename(video_path)
    result = {'filename': filename, 'transcript': "", 'summary': "", 'error': None}
    try:
        result['transcript'] = transcribe_video_file(video_path)
        key_frames = extract_key_frames(video_path, frame_limit=KEY_FRAME_LIMIT)
        if not key_frames: raise ValueError("No key frames extracted.")

        frame_b64_list = _convert_frames_to_base64(key_frames, filename)
        if not frame_b64_list: raise ValueError("No valid frames could be processed.")

        summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(transcript=result['transcript'])

        summary_response = extract_image_features_with_llm(frame_b64_list, prompt=summary_prompt, feature_gen=True)

        if summary_response and isinstance(summary_response, list):
            result['summary'] = str(summary_response[0])
        else:
            result['summary'] = "[ERROR: Invalid or empty summary response]"

    except Exception as e:
        logger.error(f"Failed to process video {filename}: {e}")
        result['error'] = str(e)
    return result


def _analyze_video_summaries(summaries: Dict[str, str], output_formats: List[str], description: str) -> Dict:
    """Performs text analysis on video summaries and returns structured data."""
    temp_summary_files = []
    os.makedirs('uploads', exist_ok=True)
    try:
        for filename, summary_text in summaries.items():
            temp_path = os.path.join('uploads', f"summary_{os.path.splitext(filename)[0]}.txt")
            with open(temp_path, 'w', encoding='utf-8') as f: f.write(summary_text)
            temp_summary_files.append(temp_path)

        text_analysis = process_text_files(temp_summary_files, output_formats, description)
        tabular_output = {}
        analysis_data = text_analysis.get('tabular_output', {})
        for path in temp_summary_files:
            original_fname = next(
                (fname for fname in summaries if os.path.splitext(fname)[0] in os.path.basename(path)), None)
            if original_fname:
                tabular_output[original_fname] = analysis_data.get(os.path.basename(path),
                                                                   {"error": "No data returned"})
        return {'tabular_output': tabular_output, 'feature_specification': text_analysis.get('feature_specification')}
    finally:
        for f in temp_summary_files:
            if os.path.exists(f): os.remove(f)


def process_video_files(file_paths: List[str], output_formats: Optional[List[str]] = None,
                        description: Optional[str] = None) -> dict:
    logger.info("Processing video files...")
    if not file_paths:
        return {'status': 'error', 'error': 'file_paths must be non-empty', 'type': 'video'}

    final_formats = [fmt.lower() for fmt in (output_formats or []) if fmt in VALID_OUTPUT_FORMATS] or ['json']
    all_transcripts, all_summaries, consolidated_output = {}, {}, {}

    for path in file_paths:
        if not (isinstance(path, str) and os.path.isfile(path)):
            fname = os.path.basename(str(path)) if isinstance(path, str) else "unknown"
            consolidated_output[fname] = {"error": f"Invalid or non-existent path: {path}"}
            continue

        result = _process_single_video(path)
        all_transcripts[result['filename']] = result['transcript']
        all_summaries[result['filename']] = result['summary']
        if result['error']: consolidated_output[result['filename']] = {"error": result['error']}

    valid_summaries = {k: v for k, v in all_summaries.items() if v and not v.startswith('[ERROR')}
    feature_spec = None
    if valid_summaries:
        analysis_result = _analyze_video_summaries(valid_summaries, final_formats, description)
        consolidated_output.update(analysis_result.get('tabular_output', {}))
        feature_spec = analysis_result.get('feature_specification')

    output_data = _format_outputs(consolidated_output, final_formats)

    return {
        'status': 'processed', 'type': 'video', 'original_files': file_paths,
        'output_formats': final_formats, 'description': description or "",
        'tabular_output': consolidated_output, 'outputs': output_data,
        'feature_specification': feature_spec,
        'transcripts': all_transcripts, 'summaries': all_summaries
    }


# --- 8. Miscellaneous Utilities ---

def select_representative_images(items: List[Any], sample_size: int) -> List[Any]:
    """Selects a random sample from a list of items."""
    if len(items) <= sample_size:
        return items
    return random.sample(items, sample_size)


def extract_key_frames(video_path: str, frame_limit: int = 8, sharpness_threshold: float = 100.0) -> List[np.ndarray]:
    """Extracts distinct and sharp key frames from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return []

    candidates, frame_skip, frame_count, prev_gray = [], 15, 0, None
    while True:
        is_read, frame = cap.read()
        if not is_read: break

        frame_count += 1
        if frame_count % frame_skip != 0: continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray_frame, cv2.CV_64F).var() < sharpness_threshold: continue

        change_score = cv2.absdiff(prev_gray, gray_frame).sum() if prev_gray is not None else 0
        candidates.append((change_score, frame))
        prev_gray = gray_frame

    cap.release()
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [frame for score, frame in candidates[:frame_limit]]


def create_collage(image_arrays: List[np.ndarray], collage_width: int = 1024) -> Optional[Image.Image]:
    """Creates a horizontal collage from a list of image arrays."""
    if not image_arrays: return None

    target_height = collage_width // len(image_arrays) if len(image_arrays) > 0 else 0
    if target_height == 0: return None

    pil_images = []
    total_width = 0
    for frame in image_arrays:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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


# --- 9. Prompt Templates ---

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