from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import os
import requests
from pydantic import BaseModel
from pathlib import Path
import New_Samples_PreProcessingMain as n_sample_pp
from Libraries import pd, json
import Pre_Processing as pp
import TRM_XAI as xai
from typing import List, Dict
from fastapi import Body
import time

app = FastAPI()

MODEL_PATH = "shared/best_model_details.joblib"  # Relative path for model


class FilePath(BaseModel):
	file_path: str


# Get the directory where this script is located`
BASE_DIR = Path(__file__).parent


@app.post("/train")
async def train_model(file_data: FilePath):
	"""Train the ML model from a CSV file on the server."""
	try:
		# Resolve relative path to absolute path
		file_path = (BASE_DIR / file_data.file_path).resolve()
		print(f"Looking for file at: {file_path}")  # Debugging
		
		if not file_path.exists():
			# List directory contents for debugging
			print(f"Contents of {BASE_DIR}: {list(BASE_DIR.iterdir())}")
			raise HTTPException(status_code=400, detail=f"File not found: {file_path}")
		"""
		do staff
				# Read the CSV file
		df = pd.read_csv(file_path)
		
		# Example model: mean of numeric columns
		model = df.mean().to_dict()
		joblib.dump(model, MODEL_PATH)
		
		"""
		
		return {"status": "Training completed and model saved."}
	
	except Exception as e:
		print(f"Error during training: {str(e)}")
		raise HTTPException(status_code=500, detail=f"Error during training: {str(e)}")


@app.post("/process")
async def process(input_data: List[Dict] = Body(...)):
	try:
		if not os.path.exists(MODEL_PATH):
			raise HTTPException(status_code=400, detail="Model not trained yet.")
		
		# input_data is already a list of dicts
		data = pd.DataFrame(input_data)
		data_dict = data.to_dict('records')
		start_ml = time.time()
		processed_data_dict, pre_preprocessing_dict, model_dict = n_sample_pp.preprocessing_new_samples(
			data=data,
			processing_joblib_path=os.path.join(os.path.dirname(__file__), 'shared', 'preprocessing_input'),
			model_joblib_path=os.path.join(os.path.dirname(__file__), 'shared', 'best_model_details')
		)
		end_ml = time.time()
		elapsed_ml = end_ml - start_ml
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
			f"Elapsed time for ML Part: {weeks} weeks, {days} days, {hours} hours, {minutes} minutes, {seconds} seconds, {milliseconds} milliseconds")
		star_xai = time.time()
		xai_report = xai.explainability(model_dict, pre_preprocessing_dict, processed_data_dict, data_dict)
		end_xai = time.time()
		elapsed_xai = end_xai - star_xai
		# Calculate weeks, days, hours, minutes, seconds, and milliseconds
		weeks = int(elapsed_xai // (7 * 24 * 3600))  # 7 days in a week
		elapsed_xai %= (7 * 24 * 3600)  # Remaining seconds after extracting weeks
		
		days = int(elapsed_xai // (24 * 3600))
		elapsed_xai %= (24 * 3600)  # Remaining seconds after extracting days
		
		hours = int(elapsed_xai // 3600)
		elapsed_xai %= 3600  # Remaining seconds after extracting hours
		
		minutes = int(elapsed_xai // 60)
		elapsed_xai %= 60  # Remaining seconds after extracting minutes
		
		seconds = int(elapsed_xai)
		milliseconds = int((elapsed_xai - seconds) * 1000)
		# Print the result
		print(
			f"Elapsed time for XAI Part: {weeks} weeks, {days} days, {hours} hours, {minutes} minutes, {seconds} seconds, {milliseconds} milliseconds")
		
		return xai_report
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Error in ML pipeline: {str(e)}")


if __name__ == "__main__":
	import uvicorn
	
	uvicorn.run(
		app,
		host="127.0.0.2",
		port=8000,
		reload=False,  # disable reload for debugging
		log_level="debug"
	)
