#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import time
from datetime import datetime
import logging
import config  # Assuming config contains JSONL_PATH, RESULTS_PATH
from openai import OpenAI

client = OpenAI(api_key=config.API_KEY)

# Set up logging
logging.basicConfig(level=logging.INFO)


# %%


import logging
import time
from datetime import datetime

def create_and_monitor_batch_job(model_name, jsonl_file_path, results_file_path, wait_time_minutes=3):
    # Upload the batch file
    try:
        batch_input_file = client.files.create(
            file=open(jsonl_file_path, "rb"),
            purpose="batch"
        )
        file_id = batch_input_file.id
        logging.info(f"Batch file ID: {file_id}")
    except Exception as e:
        logging.error(f"Error uploading batch file: {e}")
        return

    # Create the batch testing job
    try:
        batch_input_file_id = file_id

        batch_job = client.batches.create(
            input_file_id=batch_input_file_id,
            completion_window="24h",
            endpoint="/v1/chat/completions",
            metadata={
                "description": "active learning run"
            }
        )
        job_id = batch_job.id
        logging.info(f"Batch job ID: {job_id}")
    except Exception as e:
        logging.error(f"Error creating batch job: {e}")
        return

    # Monitor job status
    while True:
        try:
            job_status = client.batches.retrieve(job_id)
            status = job_status.status

            if status in ['completed', 'failed']:
                logging.info(f"Job {status} at {datetime.now().isoformat()}")
                
                # Download the results if succeeded
                if status == 'succeeded':
                    try:
                        results_file_url = job_status.result_file_url
                        response = client.files.download(results_file_url)
                        with open(results_file_path, 'wb') as results_file:
                            results_file.write(response.content)
                        logging.info(f"Results have been saved to {results_file_path}")
                    except Exception as e:
                        logging.error(f"Error downloading results: {e}")
                
                return job_id  # Return the job ID of the completed/failed job

            logging.info(f"Job status: {status} at {datetime.now().isoformat()}")
            time.sleep(wait_time_minutes * 60)
        except Exception as e:
            logging.error(f"Error retrieving job status: {e}")
            return


# %%


def main():
    # Example values; replace these with actual paths and model names
    jsonl_file_path = config.TESTING_JSONL_PATH
    ## results_file_path = config.RESULTS_PATH
    model_name = config.BATCH_RESULTS_PATH
    
    create_and_monitor_batch_job(model_name, jsonl_file_path, results_file_path)

if __name__ == "__main__":
    main()

