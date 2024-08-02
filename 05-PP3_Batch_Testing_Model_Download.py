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
import logging
import sys


# In[3]:


def get_latest_batch_id():
    model_names_file = os.path.join(os.getcwd(), config.FINE_TUNED_MODEL_NAME)
    df = pd.read_csv(model_names_file)
    
    # Filter the column 'batch_id' for non-empty values that start with 'batch_'
    filtered_df = df[df['batch_id'].str.startswith('batch_', na=False)].dropna(subset=['batch_id'])
    
    # Return the bottom-most row's 'batch_id'
    if not filtered_df.empty:
        return filtered_df['batch_id'].iloc[-1]
    else:
        return None



# In[4]:


def retrieve_batch_output(batch_id):
    """
    Retrieve and parse the output file of a batch job from OpenAI.

    Parameters:
    - api_key (str): The API key for authenticating with OpenAI.
    - batch_id (str): The ID of the batch job to retrieve.

    Returns:
    - List[dict]: A list of parsed JSON objects from the batch output file.
    """
    client = openai.OpenAI(api_key=config.API_KEY)

    # Retrieve job status
    job_status = client.batches.retrieve(batch_id)
    # print(job_status)

    # Get the output file content
    batch_output_file = (client.files.content(job_status.output_file_id)).read().decode('utf-8')

    # Print the content to inspect it
    # print(batch_output_file)

    # Parse JSON data
    json_data = []
    for line in batch_output_file.splitlines():
        try:
            json_data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")

    return json_data


# In[5]:


def extract_content_fields(content):
    try:
        if content and isinstance(content, list):
            item = content[0]  # Assuming there's only one item in the list
            message_content = item.get('message', {}).get('content', '')
            logprobs = item.get('logprobs', {}).get('content', [])
            
            tokens = []
            logprob_values = []
            top_logprobs_tokens = []
            top_logprobs_values = []
            
            for logprob in logprobs:
                tokens.append(logprob.get('token', ''))
                logprob_values.append(logprob.get('logprob', None))
                
                for top_logprob in logprob.get('top_logprobs', []):
                    top_logprobs_tokens.append(top_logprob.get('token', ''))
                    top_logprobs_values.append(top_logprob.get('logprob', None))
            
            return pd.Series({
                'message_content': message_content,
                'tokens': tokens,
                'logprob_values': logprob_values,
                'top_logprobs_tokens': top_logprobs_tokens,
                'top_logprobs_values': top_logprobs_values
            })
        else:
            return pd.Series({
                'message_content': '',
                'tokens': [],
                'logprob_values': [],
                'top_logprobs_tokens': [],
                'top_logprobs_values': []
            })
    except Exception as e:
        print(f"Error parsing content: {e}")
        return pd.Series({
            'message_content': '',
            'tokens': [],
            'logprob_values': [],
            'top_logprobs_tokens': [],
            'top_logprobs_values': []
        })


# In[6]:


def main():
    # Define locations
    predictions_file = os.path.join(os.getcwd(), config.PREDICTIONS_PATH)
    
    # Initialize the OpenAI client
    client = openai.OpenAI(api_key=config.API_KEY)
    
    # Get most recent batch file
    batch_job_id = get_latest_batch_id() 
    
    # Retrieve the status of the fine-tuning job
    batch_id = client.batches.retrieve(batch_job_id)

    # Get the file content
    batch_response = retrieve_batch_output(batch_id.id)

    # Normalize JSON data into a DataFrame
    batch_response_df = pd.json_normalize(batch_response)

    # Apply the extraction function to the 'response.body.choices' field
    batch_content_df = batch_response_df['response.body.choices'].apply(lambda x: extract_content_fields(x))

    # Concatenate the extracted content fields with the main DataFrame
    batch_final_df = pd.concat([batch_response_df, batch_content_df], axis=1)

    # Drop the original 'response.body.choices' column if it's no longer needed
    batch_final_df = batch_final_df.drop(columns=['response.body.choices'])

    # Save the DataFrame to a CSV file
    batch_final_df.to_csv(predictions_file, index=False)

    print ("Done!")


# In[7]:


if __name__ == "__main__":
    main()


# In[ ]:




