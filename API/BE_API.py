from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
import json
import pandas as pd
from pathlib import Path
import time

app = FastAPI()

# ML_PIPELINE_URL = "http://127.0.0.2:8000/train"
# ML_PROCESS_URL = "http://127.0.0.2:8000/process"
# DECISION_ENGINE_URL = "http://127.0.0.2:8001"

ML_PIPELINE_URL = "http://ml_service:8000/train"
ML_PROCESS_URL = "http://ml_service:8000/process"
DECISION_ENGINE_URL = "http://de_service:8001"

# Get the directory where this script is located
BASE_DIR = Path(__file__).parent


class DecisionInput(BaseModel):
	ml_input: list
	rules: list


@app.post("/start-training")
async def start_training():
	"""Train ML model using the CSV file path provided."""
	
	"""try:
		# Send file path to ML API
		response = requests.post(ML_PIPELINE_URL, json=file_data.dict())
		return response.json()
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))"""


@app.post("/run-full-decision")
async def run_full_decision(input_data: DecisionInput):
	"""
	Full decision flow:
	- Receive feature input (ml_input) and rules via request body.
	- Send features to ML_API for prediction.
	- Send ML prediction and features to DE_API for final decision using rules.
	"""
	# Unpack the received data
	input = input_data.ml_input
	rules = input_data.rules
	
	print("Received Features (ml_input):", input)
	print("Received Rules:", rules)
	try:
		full_start = time.time()
		ml_start = time.time()
		# --- Step 1: Send features to ML_API ---
		ml_response = requests.post(ML_PROCESS_URL, json=input)
		
		if ml_response.status_code != 200:
			# Check for JSON in response detail, or use raw text
			detail_text = ml_response.json().get("detail") if ml_response.text else "Unknown error"
			raise HTTPException(status_code=ml_response.status_code, detail=f"ML processing failed: {detail_text}")
		
		ml_output = ml_response.json()
		print("ML Output:", ml_output)
		ml_output = json.loads(ml_output)
		ml_end = time.time()
		# Calculate the elapsed time
		elapsed_ml = ml_end - ml_start
		# Calculate weeks, days, hours, minutes, seconds, and milliseconds
		weeks = int(elapsed_ml // (7 * 24 * 3600))  # 7 days in a week
		elapsed_ml %= (7 * 24 * 3600)  # Remaining seconds after extracting weeks
		
		days = int(elapsed_ml // (24 * 3600))
		elapsed_ml %= (24 * 3600)  # Remaining seconds after extracting days
		
		hours = int(elapsed_ml // 3600)
		elapsed_ml %= 3600  # Remaining seconds after extracting hours
		
		minutes = int(elapsed_ml // 60)
		elapsed_ml %= 60  # Remaining seconds after extracting minutes
		
		seconds = int(elapsed_ml)
		milliseconds = int((elapsed_ml - seconds) * 1000)
		
		# Print the result
		print(
			f"Elapsed time for ML Pipeline: {weeks} weeks, {days} days, {hours} hours, {minutes} minutes, {seconds} seconds, {milliseconds} milliseconds")
		# --- Step 2: Retrieve final decision from DE_API ---
		# Combine the data into a single dictionary expected by the DE API
		combined_data = {
			"ml_output": ml_output,
			"trm_features": rules  # 'rules' are the original transaction rules
		}
		de_start = time.time()
		result_response = requests.post(
			f"{DECISION_ENGINE_URL}/decision",
			json=combined_data
		)
		
		if result_response.status_code != 200:
			# Propagate any error from the Decision Engine
			detail_text = result_response.json().get("detail") if result_response.text else "Unknown error"
			raise HTTPException(
				status_code=result_response.status_code,
				detail=f"Decision Engine failed: {detail_text}"
			)
		print(result_response.json())
		
		de_end = time.time()
		# Calculate the elapsed time
		elapsed_de = de_end - de_start
		# Calculate weeks, days, hours, minutes, seconds, and milliseconds
		weeks = int(elapsed_de // (7 * 24 * 3600))  # 7 days in a week
		elapsed_de %= (7 * 24 * 3600)  # Remaining seconds after extracting weeks
		
		days = int(elapsed_de // (24 * 3600))
		elapsed_de %= (24 * 3600)  # Remaining seconds after extracting days
		
		hours = int(elapsed_de // 3600)
		elapsed_de %= 3600  # Remaining seconds after extracting hours
		
		minutes = int(elapsed_de // 60)
		elapsed_de %= 60  # Remaining seconds after extracting minutes
		
		seconds = int(elapsed_de)
		milliseconds = int((elapsed_de - seconds) * 1000)
		
		# Print the result
		print(
			f"Elapsed time for Decision Part: {weeks} weeks, {days} days, {hours} hours, {minutes} minutes, {seconds} seconds, {milliseconds} milliseconds")
		full_end = time.time()
		
		elapsed_full = full_end - full_start
		# Calculate weeks, days, hours, minutes, seconds, and milliseconds
		weeks = int(elapsed_full // (7 * 24 * 3600))  # 7 days in a week
		elapsed_full %= (7 * 24 * 3600)  # Remaining seconds after extracting weeks
		
		days = int(elapsed_full // (24 * 3600))
		elapsed_full %= (24 * 3600)  # Remaining seconds after extracting days
		
		hours = int(elapsed_full // 3600)
		elapsed_full %= 3600  # Remaining seconds after extracting hours
		
		minutes = int(elapsed_full // 60)
		elapsed_full %= 60  # Remaining seconds after extracting minutes
		
		seconds = int(elapsed_full)
		milliseconds = int((elapsed_full - seconds) * 1000)
		
		# Print the result
		print(
			f"Elapsed time for all trm: {weeks} weeks, {days} days, {hours} hours, {minutes} minutes, {seconds} seconds, {milliseconds} milliseconds")
		print(result_response.json())
		return result_response.json()
	
	except requests.exceptions.RequestException as req_e:
		# Handle connection errors to other services
		raise HTTPException(
			status_code=503,
			detail=f"Service Unavailable: Could not connect to an external API. Error: {str(req_e)}"
		)
	except Exception as e:
		# Handle all other unexpected errors
		raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
	import uvicorn
	
	uvicorn.run(
		app,
		host="127.0.0.2",
		port=8002,
		reload=False,  # disable reload for debugging
		log_level="debug"
	)
