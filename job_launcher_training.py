# +
import time
from datetime import datetime
import logging
import config  # Assuming config contains TUNING_JSONL_PATH and VALIDATION_JSONL_PATH
from openai import OpenAI

client = OpenAI(api_key=config.API_KEY)

# Set up logging
logging.basicConfig(level=logging.INFO)

def upload_file(file_path, purpose="fine-tune"):
    try:
        with open(file_path, "rb") as file:
            file_response = client.files.create(file=file, purpose=purpose)
        file_id = file_response.id
        logging.info(f"Uploaded file ID: {file_id}")
        return file_id
    except Exception as e:
        logging.error(f"Error uploading file {file_path}: {e}")
        return None

def create_fine_tuning_job(model_name, training_file_id, validation_file_id):
    try:
        job_response = client.fine_tuning.jobs.create(
            training_file=training_file_id,
            validation_file=validation_file_id,
            model=model_name
        )
        job_id = job_response.id
        logging.info(f"Fine-tune job ID: {job_id}")
        return job_id
    except Exception as e:
        logging.error(f"Error creating fine-tuning job: {e}")
        return None

def monitor_job(job_id, wait_time_minutes=3):
    while True:
        try:
            job_status = client.fine_tuning.jobs.retrieve(job_id)
            status = job_status.status

            logging.info(f"Job status: {status} at {datetime.now().isoformat()}")

            if status in ['succeeded', 'failed']:
                logging.info(f"Job {status} at {datetime.now().isoformat()}")
                return job_id  # Return the job ID of the completed/failed job

            time.sleep(wait_time_minutes * 60)
        except Exception as e:
            logging.error(f"Error retrieving job status: {e}")
            return None

def create_and_monitor_job(model_name, wait_time_minutes=3):
    # Upload the fine-tuning and validation files
    tuning_file_id = upload_file(config.TUNING_JSONL_PATH)
    validation_file_id = upload_file(config.VALIDATION_JSONL_PATH)

    if not tuning_file_id or not validation_file_id:
        return

    # Create the fine-tuning job
    job_id = create_fine_tuning_job(model_name, tuning_file_id, validation_file_id)
    if not job_id:
        return

    # Monitor the job status
    return monitor_job(job_id, wait_time_minutes)

# Example usage:
# create_and_monitor_job("model-name")

