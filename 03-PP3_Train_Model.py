#!/usr/bin/env python
# coding: utf-8

# In[1]:


### This program runs the current training set and validation through the API to get a new model


# In[2]:


import pandas as pd
from datetime import datetime
import base64
from io import StringIO
import json
import openai
from openai import OpenAI
import os
import config
from JSONL_maker_training import generate_jsonl_for_finetuning
from job_launcher_training import create_and_monitor_job
import logging
import sys


# In[3]:


import os
import pandas as pd
from datetime import datetime
import config

def get_model_name():
    fine_tuned_model_path = config.FINE_TUNED_MODEL_NAME
    
    # Define the default model and date
    default_model = "gpt-3.5-turbo-0125"
    default_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Check if the CSV file exists
    if os.path.exists(fine_tuned_model_path):
        # Load it into a DataFrame
        df = pd.read_csv(fine_tuned_model_path)
        
        # Check for a row where 'model' is not empty and both 'output' and 'log_file' are empty
        matching_row = df[(df['model'].notna()) & (df['output'].isna()) & (df['log_file'].isna())]
        
        if not matching_row.empty:
            return matching_row.iloc[0]['model']
        
        # Ensure the DataFrame has a 'date' column and is not empty before proceeding
        if 'date' in df.columns and not df.empty:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            
            if not df.empty:
                most_recent_row = df.loc[df['date'].idxmax()]
                return most_recent_row['output']
            else:
                return default_model
        else:
            return default_model
    else:
        # Create a new DataFrame with the required structure
        new_df = pd.DataFrame({
            "model": [default_model],
            "date": [default_date],
            "output": [None],
            "log_file": [None],
            "job_id": [None]
        })
        new_df.to_csv(fine_tuned_model_path, index=False)
        return default_model


# In[4]:


def log_finetuning_job_status(ft_job_id):
    print(f"Retrieving status for job ID: {ft_job_id}")
    # Initialize the OpenAI client
    client = openai.OpenAI(api_key=config.API_KEY)
    
    # Retrieve the status of the fine-tuning job
    job_status = client.fine_tuning.jobs.retrieve(ft_job_id)
    
    # Prepare the log data
    log_data = {
        'id': job_status.id,
        'created_at': job_status.created_at,
        'error': {
            'code': job_status.error.code,
            'message': job_status.error.message,
            'param': job_status.error.param
        } if job_status.error else None,
        'fine_tuned_model': job_status.fine_tuned_model,
        'finished_at': job_status.finished_at,
        'hyperparameters': {
            'n_epochs': job_status.hyperparameters.n_epochs,
            'batch_size': job_status.hyperparameters.batch_size,
            'learning_rate_multiplier': job_status.hyperparameters.learning_rate_multiplier
        },
        'model': job_status.model,
        'object': job_status.object,
        'organization_id': job_status.organization_id,
        'result_files': job_status.result_files,
        'seed': job_status.seed,
        'status': job_status.status,
        'trained_tokens': job_status.trained_tokens,
        'training_file': job_status.training_file,
        'validation_file': job_status.validation_file,
        'estimated_finish': job_status.estimated_finish,
        'integrations': job_status.integrations,
        'user_provided_suffix': job_status.user_provided_suffix
    }
    
    # Retrieve the performance data from the result files
    performance_data = []
    for file_id in job_status.result_files:
        try:
            # Retrieve the content of the file
            response = client.files.content(file_id)
            
            # Read the binary content
            content = response.read()
            
            # Decode the content assuming it is base64 encoded
            decoded_content = base64.b64decode(content).decode('utf-8')
            
            # Convert the content into a pandas DataFrame
            df = pd.read_csv(StringIO(decoded_content), delimiter=',')  # Adjust delimiter if necessary
            
            # Extract performance data (assuming specific columns; adjust as needed)
            performance_data.append(df)
            
        except Exception as e:
            print(f"Failed to retrieve or decode content for file {file_id}: {e}")
    
    # Log performance data if any
    if performance_data:
        log_data['performance_data'] = [df.to_dict(orient='records') for df in performance_data]
    else:
        log_data['performance_data'] = None
    
    # Create a logs directory if it doesn't exist
    logs_dir = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Create a log file with the current timestamp
    log_file = os.path.join(logs_dir, f"finetuning_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    # Write the log data to the log file
    with open(log_file, 'w') as file:
        json.dump(log_data, file, indent=4)
    
    print(f"Fine-tuning job status and performance data logged in {log_file}")
    return log_file


# In[5]:


import pandas as pd
import os
import config

def update_fine_tuned_model_file(log_data, log_file):
    print("Updating fine-tuned model file...")
    # Load the CSV file into a DataFrame
    fine_tuned_model_path = config.FINE_TUNED_MODEL_NAME
    
    if not os.path.exists(fine_tuned_model_path):
        print(f"Error: The file {fine_tuned_model_path} does not exist. Halting the program.")
        return

    df = pd.read_csv(fine_tuned_model_path)

    # Ensure columns are of correct data types
    df['output'] = df['output'].astype('object')
    df['log_file'] = df['log_file'].astype('object')
    df['job_id'] = df['job_id'].astype('object')

    # Extract necessary values from log_data
    model_name = log_data.model
    finished_at = log_data.finished_at
    fine_tuned_model = log_data.fine_tuned_model
    job_id = log_data.id
    
    # Find a row where the value of 'model' matches the value of 'model_name' and 'output' and 'log_file' are empty
    matching_rows = df[(df['model'] == model_name) & (df['output'].isna()) & (df['log_file'].isna())]
    
    if not matching_rows.empty:
        # Update the first matching row
        df.loc[matching_rows.index[0], 'date'] = finished_at
        df.loc[matching_rows.index[0], 'output'] = fine_tuned_model
        df.loc[matching_rows.index[0], 'log_file'] = log_file
        df.loc[matching_rows.index[0], 'job_id'] = job_id
    else:
        # Concatenate a new row to the DataFrame
        new_row = {
            "model": model_name,
            "date": finished_at,
            "output": fine_tuned_model,
            "log_file": log_file,
            "job_id": job_id
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save the updated DataFrame back to the CSV file
    df.to_csv(fine_tuned_model_path, index=False)
    print("The fine_tuning_model_name file has been updated")



# In[6]:


def validation_data():   
    validation_output = os.path.join(os.getcwd(), config.VALIDATION_JSONL_PATH)
    label_validation_data_path = os.path.join(os.getcwd(), config.LABEL_VALIDATION_SET_PATH)
    
    # Check if the JSONL file already exists
    if os.path.exists(validation_output):
        print(f"JSONL file already exists at {validation_output}. Returning to main loop.")
        return

    # If the JSONL file does not exist, load the validation data and generate the JSONL file
    validation_data = pd.read_csv(label_validation_data_path)
    
    generate_jsonl_for_finetuning(validation_data, validation_output)
    
    return


# In[7]:


def main(model_name):
    # Define locations
    output_file = os.path.join(os.getcwd(), config.TUNING_JSONL_PATH)
    training_data_path = os.path.join(os.getcwd(), config.CURRENT_TRAINING_SET_PATH)
    
    validation_data()
        
    # Read the data into DataFrames
    training_data = pd.read_csv(training_data_path)    
    
    print(f"Starting model is: {model_name}")
    
    # Call the generate_jsonl_for_finetuning function
    generate_jsonl_for_finetuning(training_data, output_file)

    # Call the create_and_monitor_job function
    ft_job_id = create_and_monitor_job(model_name, 5)

    print (ft_job_id)
    
    # Initialize the OpenAI client
    client = openai.OpenAI(api_key=config.API_KEY)
    
    # Retrieve the status of the fine-tuning job
    job_status = client.fine_tuning.jobs.retrieve(ft_job_id)

    # Log the fine-tuning job status and performance data
    log_file = log_finetuning_job_status(ft_job_id)
    
    # Update the fine-tuned model CSV file with the job details
    update_fine_tuned_model_file(job_status, log_file)



# In[8]:


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




