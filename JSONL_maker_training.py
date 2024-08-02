#!/usr/bin/env python
# coding: utf-8
# %%
# #!/usr/bin/env python
# coding: utf-8
import config
import json
import pandas as pd
import os

def load_prompt_template(prompt_file_path):
    """
    Load the prompt template from a JSON file.
    """
    try:
        with open(prompt_file_path, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Error loading {prompt_file_path}: {e}")

def load_categories(categories_file_path):
    """
    Load the categories from a CSV file.
    """
    try:
        categories_df = pd.read_csv(categories_file_path)
        return ', '.join(categories_df['label'].tolist())
    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError) as e:
        raise ValueError(f"Error loading {categories_file_path}: {e}")

def generate_jsonl_for_finetuning(validation_data, output_file):
    """
    Generate a JSONL file for fine-tuning.
    """
    # validation_data_path = os.path.join(os.getcwd(), config.CURRENT_TRAINING_SET_PATH)
    # output_file = os.path.join(os.getcwd(), config.TUNING_JSONL_PATH)
    # validation_data = pd.read_csv(validation_data_path)
    
    required_columns = ['analyze_text', 'human_label']
    if not all(column in validation_data.columns for column in required_columns):
        raise KeyError(f"Validation data must contain columns: {required_columns}")

    prompt_template = load_prompt_template(config.PROMPT_TEMPLATE_TEXT)
    categories = load_categories(config.PROMPT_TEMPLATE_LABELS)

    prompt_template = [
        {k: v.replace("{categories}", categories) for k, v in entry.items()}
        for entry in prompt_template
    ]
    
    jsonl_data = [
        {
            "messages": [
                {"role": "system", "content": prompt_template[0]["content"]},
                {"role": "user", "content": prompt_template[1]['content'].replace("{train_text}", row['analyze_text'])},
                {"role": "assistant", "content": row['human_label']}
            ]
        }
        for _, row in validation_data.iterrows()
    ]

    with open(output_file, 'w') as file:
        for entry in jsonl_data:
            file.write(json.dumps(entry) + '\n')

    print(f"JSONL file '{output_file}' has been generated successfully.")

# Example usage (commented out for script inclusion)
# validation_data = pd.DataFrame({
#     'analyze_text': ["example text 1", "example text 2"],
#     'human_label': ["label 1", "label 2"]
# })
# output_file = "fine_tuning_data.jsonl"
# generate_jsonl_for_finetuning(validation_data, output_file)

