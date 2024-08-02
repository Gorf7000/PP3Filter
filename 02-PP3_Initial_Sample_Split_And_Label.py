#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This program calls up the initial data file, divides it into appropriate samples, and sends it out for labeling

import pandas as pd
from sklearn.model_selection import train_test_split
import config
from query_oracle import query_oracle

def split_data():
    """
    Splits the dataset into three parts: initial training set, validation set, 
    and active learning set. The ratios for splitting are defined within the function.
    
    The splits are saved to CSV files specified in the config file.
    """
    # Read the raw dataset from the specified path
    try:
        text_df = pd.read_csv(config.RAW_DATA_SET_PATH)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error: The file at {config.RAW_DATA_SET_PATH} was not found.") from e
    except pd.errors.EmptyDataError as e:
        raise ValueError("Error: The file is empty.") from e
    except Exception as e:
        raise RuntimeError(f"An error occurred while reading the dataset: {str(e)}") from e

    # Calculate the fixed number of entries based on the specified ratios
    total_count = len(text_df)
    initial_train_count = int(total_count * 0.25)  # 25% for initial training
    validation_count = int(total_count * 0.15)  # 15% for validation

    # Ensure there are enough rows to split
    if initial_train_count + validation_count > total_count:
        raise ValueError("Error: The sum of initial_train_count and validation_count exceeds the total number of rows in the dataset.")

    # Split the dataset into initial training set and remaining set
    initial_train_df, remaining_df = train_test_split(text_df, train_size=initial_train_count, random_state=42)

    # Split the remaining set into validation set and active learning set
    active_learning_df, validation_df = train_test_split(remaining_df, test_size=validation_count, random_state=42)

    # Save the splits to CSV files
    try:
        initial_train_df.to_csv(config.INITIAL_TRAINING_SET_PATH, index=False)
        validation_df.to_csv(config.VALIDATION_SET_PATH, index=False)
        active_learning_df.to_csv(config.ACTIVE_LEARNING_SET_PATH, index=False)
    except Exception as e:
        raise RuntimeError(f"An error occurred while saving the datasets: {str(e)}") from e

    print("Data splits have been saved to CSV files.")

def main():
    """
    Main program loop to split data, label initial training and validation sets, 
    and save the labeled data to specified paths.
    """
    # Split the dataset into initial training, validation, and active learning sets
    split_data()
    
    # Load the initial training set
    initial_train_df = pd.read_csv(config.INITIAL_TRAINING_SET_PATH)
    # Run query_oracle on the initial training set for manual labeling
    labeled_initial_train_df = query_oracle(initial_train_df)
    # Save the labeled initial training set
    labeled_initial_train_df.to_csv(config.CURRENT_TRAINING_SET_PATH, index=False)
    
    # Load the validation set
    validation_df = pd.read_csv(config.VALIDATION_SET_PATH)
    # Run query_oracle on the validation set for manual labeling
    labeled_validation_df = query_oracle(validation_df)
    # Save the labeled validation set
    labeled_validation_df.to_csv(config.LABEL_VALIDATION_SET_PATH, index=False)
    
    print("Data processing and labeling complete. Labeled data saved to CSV files.")

if __name__ == "__main__":
    main()


# In[ ]:




