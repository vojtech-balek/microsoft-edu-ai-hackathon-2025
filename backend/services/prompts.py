from string import Template
import json
import re

image_feature_extraction_prompt = Template("""
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
            "All features should be made with respect to the user-provided target variable from the description."
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


text_feature_extraction_prompt = Template("""
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

image_feature_generation_prompt = Template("""
{
    "system_message": "IMPORTANT: Return only a valid JSON object with no explanations, text, or markdown!!! Do not include any commentary or introductory text!!!You are a feature extraction assistant for text documents.",
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

def generate_feature_queries(features, text=None, image=None):
    queries = {
        "task": "Extract the following features as described below and return a valid JSON object.",
        "constraints": [
            "The output must be a valid JSON.",
            "All answers must be simple and correspond to categorical values only."
        ],
        "features": [
            {
                "feature_name": feature['feature_name'],
                "description": feature['description'],
                "extraction_query": feature['extraction_query'].replace('{text}', text)
            } for feature in features['features']
        ],
        "output_format": {
            "type": "json",
            "structure": {
                    "<feature1 name>": "<feature1 value>"
            }
        }
    }
    if text:
        queries["input_text"] = text
    elif image:
        queries[""]

    return json.dumps(queries)


def extract_json(response: str):
    # Find JSON between ```json and ```
    match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Chyba při dekódování JSON: {e}")
    else:
        return json.loads(response.strip('```json').strip('```'))

def clean_json_output(response_dict):
    cleaned_dict = {}
    for key, value in response_dict.items():
        value = extract_json(value)
        cleaned_dict[key] = value
    return cleaned_dict