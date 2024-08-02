#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import config
import os
import ast
import numpy as np
import re
import tiktoken

# Load label tokens once and reuse
def load_label_tokens(csv_path):
    """ Load labels from the CSV file and generate tokens using GPT tokenizer. """
    labels_df = pd.read_csv(csv_path)
    labels = labels_df['label'].tolist()
    tokens_map = {label: gpt_tokenizer(label) for label in labels}
    return tokens_map

def gpt_tokenizer(text):
    """Tokenize the input text using OpenAI's GPT-3 tokenizer."""
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(text)
    token_strings = [enc.decode([token]) for token in tokens]
    return token_strings

def convert_to_list(value):
    """ Convert the string representation of a list to a list, or return the value if it's already a list. """
    if isinstance(value, str):
        try:
            return eval(value)
        except (SyntaxError, NameError):
            return []
    elif isinstance(value, list):
        return value
    else:
        return []

def validate_data(row, debug=False):
    """ Validate the row of data for required fields and format. """
    required_fields = ['tokens', 'logprob_values', 'top_logprobs_tokens', 'top_logprobs_values']
    
    for field in required_fields:
        if field not in row:
            if debug:
                print(f"Missing required field: {field}")
            return False
    
    tokens = convert_to_list(row['tokens'])
    logprob_values = convert_to_list(row['logprob_values'])
    top_logprobs_tokens = convert_to_list(row['top_logprobs_tokens'])
    top_logprobs_values = convert_to_list(row['top_logprobs_values'])
    
    if not (tokens and logprob_values and top_logprobs_tokens and top_logprobs_values):
        if debug:
            print(f"Invalid data format in row: {row}")
        return False
    
    return True


# In[2]:


def calculate_label_probabilities_per_row(row, tokens_map):
    if not validate_data(row):
        return {'custom_id': row.get('custom_id', None)}  # Return custom_id if data is invalid
    
    tokens = convert_to_list(row['tokens'])
    logprob_values = convert_to_list(row['logprob_values'])
    top_logprobs_tokens = convert_to_list(row['top_logprobs_tokens'])
    top_logprobs_values = convert_to_list(row['top_logprobs_values'])
    
    row_probabilities = {'custom_id': row.get('custom_id', None)}
    token_log_probs = dict(zip(tokens, logprob_values))
    for token, logprob in zip(top_logprobs_tokens, top_logprobs_values):
        if token not in token_log_probs:
            token_log_probs[token] = logprob
        else:
            token_log_probs[token] = max(token_log_probs[token], logprob)  # Use the highest log probability
    
    aggregated_log_probs = {label: -float('inf') for label in tokens_map}
    
    for label, label_token_ids in tokens_map.items():
        if not label_token_ids:
            continue
        
        valid_log_probs = [token_log_probs.get(token, -float('inf')) for token in label_token_ids]
        valid_log_probs = [log_prob for log_prob in valid_log_probs if log_prob > -float('inf')]
        
        if valid_log_probs:
            log_prob_sum = np.sum(valid_log_probs)
            aggregated_log_probs[label] = log_prob_sum
    
    max_log_prob = max(aggregated_log_probs.values())
    exp_sums = [np.exp(aggregated_log_probs[label] - max_log_prob) for label in aggregated_log_probs if aggregated_log_probs[label] > -float('inf')]
    total_exp_sum = sum(exp_sums)
    
    for label in aggregated_log_probs:
        if aggregated_log_probs[label] > -float('inf'):
            row_probabilities[label] = np.exp(aggregated_log_probs[label] - max_log_prob) / total_exp_sum
        else:
            row_probabilities[label] = 0.0
    
    return row_probabilities


# In[3]:


def calculate_label_probabilities(df, tokens_map):
    results = []
    
    for index, row in df.iterrows():
        try:
            row_probabilities = calculate_label_probabilities_per_row(row, tokens_map)
            results.append(row_probabilities)
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            results.append({'custom_id': row.get('custom_id', None)})
    
    results_df = pd.DataFrame(results)
    return results_df


# In[4]:


import pandas as pd

def add_calculated_results_and_save(batch_final_df, results_df, save_path):
    """Add the calculated results of each row to the correct row in batch_final_df and save the updated DataFrame."""
    
    # Ensure both DataFrames have the 'custom_id' column
    if 'custom_id' not in batch_final_df.columns or 'custom_id' not in results_df.columns:
        raise ValueError("Both DataFrames must contain the 'custom_id' column.")
    
    # Merge DataFrames on 'custom_id'
    merged_df = pd.merge(batch_final_df, results_df, on='custom_id', how='outer', suffixes=('', '_result'))
    
    # Identify columns to check for unmatched rows
    batch_final_check_column = 'Lists'
    results_check_column = 'Prose'

    # Check for rows with missing 'custom_id' in either DataFrame
    unmatched_in_batch_final = merged_df[merged_df[results_check_column].isna()]
    unmatched_in_results = merged_df[merged_df[batch_final_check_column].isna()]
    
    if not unmatched_in_batch_final.empty:
        print(f"Warning: Rows in batch_final_df not matched in results_df:\n{unmatched_in_batch_final[[batch_final_check_column]]}")
    
    if not unmatched_in_results.empty:
        print(f"Warning: Rows in results_df not matched in batch_final_df:\n{unmatched_in_results[[results_check_column]]}")
    
    # Save the updated DataFrame to a CSV file
    merged_df.to_csv(save_path, index=False)
    print(f"Updated DataFrame saved to {save_path}")

# Example usage
# batch_final_df = pd.DataFrame(...)  # Your DataFrame with 'custom_id'
# results_df = pd.DataFrame(...)      # Your DataFrame with 'custom_id'
# save_path = 'path_to_save.csv'
# add_calculated_results_and_save(batch_final_df, results_df, save_path)


# In[5]:


# Main execution
predictions_file = os.path.join(os.getcwd(), config.PREDICTIONS_PATH)
batch_final_df = pd.read_csv(predictions_file)
csv_path = os.path.join(os.getcwd(), config.PROMPT_TEMPLATE_LABELS)
tokens_map = load_label_tokens(csv_path)

results_df = calculate_label_probabilities(batch_final_df, tokens_map)
save_path = os.path.join(os.getcwd(), config.PREDICTIONS_PATH)
add_calculated_results_and_save(batch_final_df, results_df, save_path)


# In[ ]:




