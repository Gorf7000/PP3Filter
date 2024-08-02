#!/usr/bin/env python
# coding: utf-8
# %%
### Program to make jsonl files for batch processing (testing/validation)



# !/usr/bin/env python
# coding: utf-8

import config
import json
import pandas as pd



# Define the load_prompt_template function
def load_prompt_template(prompt_file_path):
    """
    Load the prompt template from a JSON file.

    Parameters:
    - prompt_file_path (str): The path to the JSON file containing the prompt template.

    Returns:
    - list: A list of dictionaries representing the prompt template.
    """
    try:
        with open(prompt_file_path, 'r') as file:
            prompt_template = json.load(file)
        return prompt_template
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {prompt_file_path} does not exist.")
    except json.JSONDecodeError:
        raise ValueError(f"The file {prompt_file_path} contains invalid JSON.")
        


# %%
def load_categories(categories_file_path):
    """
    Load the categories from a CSV file.

    Parameters:
    - categories_file_path (str): The path to the CSV file containing the categories.

    Returns:
    - str: A comma-separated string of categories.
    """
    try:
        categories_df = pd.read_csv(categories_file_path)
        categories = ', '.join(categories_df['label'].tolist())
        return categories
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {categories_file_path} does not exist.")
    except pd.errors.EmptyDataError:
        raise ValueError(f"The file {categories_file_path} is empty or contains invalid data.")
    except KeyError:
        raise KeyError(f"The file {categories_file_path} does not contain the required 'category' column.")



# %%
import json

def generate_jsonl_for_batch_testing(testing_data, model_name, output_file):
    """
    Generate a JSONL file for batch testing with a custom JSONL schema.

    Parameters:
    - testing_data (pd.DataFrame): The testing data containing text to be analyzed.
    - model_name (str): The name of the model to use.
    - output_file (str): The path to the output JSONL file.
    """
    # Check if the required column exists in the DataFrame
    if 'analyze_text' not in testing_data.columns:
        raise KeyError("Required column 'analyze_text' not found in the testing data.")
    
    # Load the prompt template
    prompt_template = load_prompt_template(config.PROMPT_TEMPLATE_TEXT)
    
    # Load the categories
    categories = load_categories(config.PROMPT_TEMPLATE_LABELS)
    
    # Replace {categories} in the prompt template
    prompt_template = [
        {k: v.replace("{categories}", categories) for k, v in entry.items()}
        for entry in prompt_template
    ]
    
    jsonl_data = []
    for i in range(len(testing_data)):
        analyze_text = testing_data['analyze_text'].iloc[i]
        custom_id = testing_data.loc[i, 'article_ID']  # Use the article_ID from the DataFrame

        jsonl_entry = {
            "custom_id": f"request-{custom_id}",  # Use the index as part of the custom_id
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": prompt_template[0]["content"]},
                    {"role": "user", "content": prompt_template[1]['content'].replace("{train_text}", analyze_text)}
                ],
                "max_tokens": 2500,
                "temperature": 0.0,
                "logprobs": True,
                "top_logprobs": 5                
            }
        }
        
        jsonl_data.append(jsonl_entry)

    # Write all entries to the file outside the loop
    with open(output_file, 'w') as file:
        for entry in jsonl_data:
            file.write(json.dumps(entry) + '\n')

    print(f"JSONL file '{output_file}' has been generated successfully.")



# %%
def main():
    # Load testing data
    testing_data = pd.read_csv(config.ACTIVE_LEARNING_SET_PATH)
    
    # Generate JSONL file
    generate_jsonl_for_batch_testing(testing_data, config.TESTING_JSONL_PATH)

if __name__ == "__main__":
    main()


# %%
