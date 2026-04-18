from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import decision_engine_version as de
import json
from typing import List, Dict
from fastapi import Body

app = FastAPI()

stored_data = {"ml_output": None, "frontend_input": None, "decision": None}

# URL for the FD API
FD_API_URL = "http://127.0.0.2:8002"


class MLOutput(BaseModel):
	prediction: float
	confidence: float


class FrontendInput(BaseModel):
	user_action: str
	additional_info: str


"""@app.post("/decision")
async def receive_ml_output(input_data: MLOutput):
	# Store ML output and request frontend input from FD.
	stored_data["ml_output"] = input_data.dict()
	stored_data["decision"] = None
	
	# Request frontend input from FD
	try:
		# Send ML output to FD so it knows what frontend input to provide
		response = requests.post(f"{FD_API_URL}/get-frontend-input", json=input_data.dict())
		if response.status_code != 200:
			raise HTTPException(status_code=response.status_code, detail="Failed to get frontend input from FD")
		
		frontend_input = response.json()
		stored_data["frontend_input"] = frontend_input
		
		# Make a decision based on ML output and frontend input
		if frontend_input.get("user_action") == "approve" and input_data.confidence > 0.8:
			decision = {"decision": "Approve", "reason": "High confidence and user approval"}
		else:
			decision = {"decision": "Reject", "reason": "Low confidence or user disapproval"}
		
		stored_data["decision"] = decision
		return {"status": "Decision computed", "decision": decision}
	
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))"""


@app.post("/decision")
async def get_result(data: Dict = Body(...)):
	ml_list = data.get("ml_output")
	trm_list = data.get("trm_features")
	
	if ml_list is None or trm_list is None:
		raise HTTPException(status_code=400, detail="Missing 'ml_output' or 'trm_features' in request body")
	
	decisions_df, decisions_list = de.run_trm_ml_engine(trm_list, ml_list)
	decisions_list = de.convert_numpy_types(decisions_list)
	
	return decisions_list


if __name__ == "__main__":
	import uvicorn
	
	uvicorn.run(
		app,
		host="127.0.0.2",
		port=8001,
		reload=False,  # disable reload for debugging
		log_level="debug"
	)
