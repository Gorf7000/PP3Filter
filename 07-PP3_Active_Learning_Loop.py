#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import config
from query_oracle import query_oracle

# Load labels from the CSV file
def load_labels(csv_path):
    labels_df = pd.read_csv(csv_path)
    labels = labels_df['label'].tolist()  # Assuming the column name in the CSV file is 'label'
    return labels

# Function to calculate entropy
def calculate_entropy(row, labels):
    probabilities = row[labels].values
    probabilities = probabilities.astype(float)  # Ensure all values are floats
    entropy = -np.sum(probabilities * np.log1p(probabilities + 1e-10))  # Use log1p for numerical stability
    return entropy

# Function to calculate margin of confidence
def calculate_margin(row, labels):
    probabilities = row[labels].values
    probabilities = probabilities.astype(float)  # Ensure all values are floats
    sorted_probs = np.sort(probabilities)
    margin = sorted_probs[-1] - sorted_probs[-2] if len(sorted_probs) > 1 else sorted_probs[-1]
    return margin

# Normalize function
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

# Calculate combined score
def calculate_combined_score(df, weights=None):
    # Default weights if not provided
    if weights is None:
        weights = {'entropy': 1.0, 'margin': 1.0, 'max_prob': 1.0}
    
    # Normalize the measures
    df['normalized_entropy'] = normalize(df['entropy'])
    df['normalized_margin'] = normalize(1 - df['margin'])  # Inverse margin, lower margin means higher uncertainty
    df['normalized_max_prob'] = normalize(1 - df['max_prob'])  # Inverse max_prob, lower max_prob means higher uncertainty
    
    # Calculate the combined score
    df['combined_score'] = (
        weights['entropy'] * df['normalized_entropy'] +
        weights['margin'] * df['normalized_margin'] +
        weights['max_prob'] * df['normalized_max_prob']
    )
    
    return df


# In[2]:


def process_predictions(weights=None):
    """
    Processes the prediction file by calculating entropy, margin, max_prob, and a combined score.
    
    Parameters:
    - weights (dict, optional): Weights for combining scores. Defaults to None, which applies equal weights.
    
    Returns:
    - DataFrame: The processed DataFrame with additional columns for entropy, margin, max_prob, and combined_score.
    """
    # Define file paths from the config file
    predictions_path = os.path.join(os.getcwd(), config.PREDICTIONS_PATH)
    labels_path = os.path.join(os.getcwd(), config.PROMPT_TEMPLATE_LABELS)
    
    # Load DataFrame
    batch_final_df = pd.read_csv(predictions_path)
    
    # Load labels
    labels = load_labels(labels_path)
    
    # Calculate entropy and margin
    batch_final_df['entropy'] = batch_final_df.apply(lambda row: calculate_entropy(row, labels), axis=1)
    batch_final_df['margin'] = batch_final_df.apply(lambda row: calculate_margin(row, labels), axis=1)
    
    # Ensure no NaN values are present before calculating max_prob
    batch_final_df[labels] = batch_final_df[labels].fillna(0)
    batch_final_df['max_prob'] = batch_final_df[labels].max(axis=1)
    
    # Calculate combined score
    if weights is None:
        weights = {'entropy': 1.0, 'margin': 1.0, 'max_prob': 1.0}
    batch_final_df = calculate_combined_score(batch_final_df, weights)
    
    # Save the processed DataFrame to a file if needed
    batch_final_df.to_csv('foo.csv', index=False)
    
    return batch_final_df


# In[3]:


def get_subset_df(df, score_threshold=3, percentage=0.15):
    """
    Gets a subset of the DataFrame by sorting and filtering based on combined_score.
    
    Parameters:
    - df (DataFrame): The DataFrame to process.
    - score_threshold (float): The combined_score threshold to filter by.
    - percentage (float): The percentage of the DataFrame to include after the threshold scores.
    
    Returns:
    - DataFrame: A new DataFrame with the rows where combined_score = score_threshold 
                 and the next percentage of the sorted DataFrame.
    """
    # Sort the DataFrame by combined_score in descending order
    sorted_df = df.sort_values(by='combined_score', ascending=False).reset_index(drop=True)
    
    # Filter rows where combined_score equals the score_threshold
    threshold_df = sorted_df[sorted_df['combined_score'] == score_threshold]
    
    # Determine the index range for the next percentage of rows
    if not threshold_df.empty:
        threshold_index = threshold_df.index[-1]
    else:
        threshold_index = -1
    
    total_rows = len(sorted_df)
    next_rows_count = int(total_rows * percentage)
    
    # Debug prints
    print(f"Total rows in sorted_df: {total_rows}")
    print(f"Total rows in threshold_df: {len(threshold_df)}")
    print(f"Number of rows to include after the threshold: {next_rows_count}")
    print(f"Threshold index: {threshold_index}")
    
    # Get the index range for the next percentage of rows
    start_index = threshold_index + 1
    end_index = min(start_index + next_rows_count, total_rows)
    
    # Ensure that the start index is valid
    if start_index < total_rows:
        subset_df = pd.concat([threshold_df, sorted_df.iloc[start_index:end_index]])
    else:
        subset_df = threshold_df
    
    # Debug print
    print(f"Start index: {start_index}")
    print(f"End index: {end_index}")
    
    # subset_df.to_csv('Subset_df.csv', index = False)
    
    return subset_df


# In[4]:


def filter_columns(df, keep_columns):
    """
    Keeps only the specified columns in the DataFrame.
    
    Parameters:
    - df (DataFrame): The DataFrame to process.
    - keep_columns (list): List of columns to keep in the DataFrame.
    
    Returns:
    - DataFrame: A new DataFrame with only the specified columns.
    """
    # Ensure the columns to keep are valid
    valid_columns = [col for col in keep_columns if col in df.columns]
    
    # Create a new DataFrame with only the specified columns
    filtered_df = df[valid_columns]
    
    # filtered_df.to_csv('FilterFoo.csv', index=False)
    
    
    return filtered_df


# In[5]:


def add_lookup_index(df, custom_id_column):
    """
    Extracts numeric values following 'request-' from the custom_id column and creates a new column 'lookup_index'.
    
    Parameters:
    - df (DataFrame): The DataFrame to process.
    - custom_id_column (str): The name of the column containing custom IDs.
    
    Returns:
    - DataFrame: The DataFrame with the new 'lookup_index' column added.
    """
    df = df.copy()
    
    # Extract numeric values following 'request-' using regular expressions
    df['lookup_index'] = df[custom_id_column].str.extract(r'request-(.+)', expand=False)
    
    # Print column names for debugging
    # print("Column names:", df.columns)
    
    return df


# In[6]:


def main():
    # Process predictions first
    processed_predictions_df = process_predictions()

    # Define file paths
    active_learning_set_path = os.path.join(os.getcwd(), config.ACTIVE_LEARNING_SET_PATH)
    add_to_training_path = os.path.join(os.getcwd(), config.LABELED_HIGH_UNCERTAIN_SET_PATH)
    current_training_path = os.path.join(os.getcwd(), config.CURRENT_TRAINING_SET_PATH)
    
    # Load the active learning DataFrame
    active_learning_df = pd.read_csv(active_learning_set_path)
    
    # Add 'lookup_index' to the processed predictions DataFrame
    if 'custom_id' in processed_predictions_df.columns:
        processed_predictions_df = add_lookup_index(processed_predictions_df, 'custom_id')
    else:
        print("Error: 'custom_id' column is missing from the processed predictions DataFrame")
        return   
    
    # Define the score threshold and percentage
    score_threshold = 3
    percentage = 0.15
    
    # Get the subset DataFrame based on combined_score
    subset_df = get_subset_df(processed_predictions_df, score_threshold, percentage)
    
    # Define the columns to keep
    keep_columns = ['combined_score', 'custom_id', 'lookup_index']
    
    # Apply the function to subset_df
    filtered_subset_df = filter_columns(subset_df, keep_columns)
    
    # Convert 'lookup_index' to a set for faster lookup
    lookup_index_set = set(filtered_subset_df['lookup_index'].dropna())
    
    # Split the DataFrame based on 'lookup_index'
    matched_df = active_learning_df[active_learning_df['article_ID'].isin(lookup_index_set)]
    not_matched_df = active_learning_df[~active_learning_df['article_ID'].isin(lookup_index_set)]
    
    # Print the number of rows in each DataFrame to verify
    print(f"Number of rows in matched_df: {len(matched_df)}")
    print(f"Number of rows in not_matched_df: {len(not_matched_df)}")
    
    # Send matched to oracle for labling
    add_to_training_df = query_oracle (matched_df)
    add_to_training_df.to_csv(add_to_training_path, index=False)
    
    # Save not matched as new active learning set
    not_matched_df.to_csv(active_learning_set_path, index=False)
    
    # Create new training set by incorporating annotated high uncertainty data points
    current_training_set_df = pd.read_csv(current_training_path)
    
    # Concatenate the two DataFrames
    combined_df = pd.concat([current_training_set_df, add_to_training_df], ignore_index=True)

    # Remove duplicate rows
    combined_df = combined_df.drop_duplicates(subset=['article_ID'])
    
    combined_df.to_csv(current_training_path, index=False)
    
    


# In[7]:


if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





# In[ ]:




