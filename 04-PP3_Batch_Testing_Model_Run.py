#!/usr/bin/env python
# coding: utf-8

# In[1]:


### This program runs the active learning set through the model


# In[2]:


import pandas as pd
from datetime import datetime
import base64
import io
from io import StringIO
import json
import openai
from openai import OpenAI
import os
import config
from JSONL_maker_batch import generate_jsonl_for_batch_testing
from job_launcher_batch import create_and_monitor_batch_job
import logging
import sys


# In[3]:


def get_model_name():
    # Load the CSV file from the location specified in config.FINE_TUNED_MODEL_NAME
    df = pd.read_csv(config.FINE_TUNED_MODEL_NAME)
    
    # Drop rows where 'output' is NaN to focus on complete rows
    df = df.dropna(subset=['output'])
    
    # Check if the DataFrame is not empty
    if not df.empty:
        # Get the last row from the DataFrame
        last_row = df.iloc[-1]
        # Return the value from the 'output' column of the last row
        model_name = last_row['output']
        return model_name
    else:
        # Return None or an appropriate value if no complete rows are found
        return None



# In[4]:


def update_batch_id(model_name, new_batch_id):
    # Load the CSV file from the location specified in config.FINE_TUNED_MODEL_NAME
    df = pd.read_csv(config.FINE_TUNED_MODEL_NAME)
    
    # Drop rows where 'output' is NaN to focus on complete rows
    df = df.dropna(subset=['output'])
    
    # Ensure the 'batch_id' column is of type str to handle string assignment
    df['batch_id'] = df['batch_id'].astype(str)
    
    # Find the index of the row where 'output' matches the model_name
    matching_rows = df[df['output'] == model_name]
    
    if not matching_rows.empty:
        # Get the index of the first matching row
        matching_row_index = matching_rows.index[0]
        
        # Update the 'batch_id' column in the matching row
        df.at[matching_row_index, 'batch_id'] = new_batch_id
        
        # Save the updated DataFrame back to the CSV file
        df.to_csv(config.FINE_TUNED_MODEL_NAME, index=False)
    else:
        # Handle the case where no matching row is found
        print(f"No row found with the model name '{model_name}'.")


# In[5]:


def main(model_name):
    # Define locations
    output_file = os.path.join(os.getcwd(), config.TESTING_JSONL_PATH)
    testing_data_path = os.path.join(os.getcwd(), config.ACTIVE_LEARNING_SET_PATH)
    
    # Read the data into DataFrames
    testing_data = pd.read_csv(testing_data_path)    
    
    # print(f"Starting model is: {model_name}")
    
    # Call the generate_jsonl_for_finetuning function
    generate_jsonl_for_batch_testing(testing_data, model_name, output_file)

    # Call the create_and_monitor_job function
    batch_job_id = create_and_monitor_batch_job(model_name, output_file, config.BATCH_RESULTS_PATH, 1)
      
    # Initialize the OpenAI client
    client = openai.OpenAI(api_key=config.API_KEY)
    
    # Retrieve the status of the fine-tuning job
    batch_id = client.batches.retrieve(batch_job_id)

    print (batch_id.id)
    update_batch_id(model_name, batch_id.id)
    print ("Done!")
    return batch_id.id
    


# In[6]:


# Prompt the user to choose between a listed model, their own model, or exit
user_choice = input("Do you want to use a listed model, provide your own, or exit? (Enter 'listed' for listed model, 'own' for your own model, or 'exit' to quit): ").strip().lower()

if user_choice == 'listed':
    model_name = get_model_name()
elif user_choice == 'own':
    model_name = input("Please enter the name of your model: ").strip()
elif user_choice == 'exit':
    print("Exiting the program.")
    import sys
    sys.exit()
else:
    print("Invalid choice. Defaulting to listed models.")
    model_name = get_model_name()

print(f"Using model: {model_name}")

if __name__ == "__main__":
    main(model_name)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




