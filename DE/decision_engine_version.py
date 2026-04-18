import logging
import numpy as np
from computing_scores import make_final_decision  ### python file containing the code for making the decision of trm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
import time
import json
import pandas as pd

# ------------------ Risk value mapping ------------------
risk_value_map = {
	"high_risk_rules": 1.0,
	"medium_risk_rules": 0.7,
	"low_medium_risk_rules": 0.4,
	"low_risk_rules": 0.1
}


# ------------------ Run Engine Function ------------------ #
def run_trm_ml_engine(trm_list, ml_list):
	import pandas as pd
	
	# Convert lists to DataFrames
	df_trm = pd.DataFrame(trm_list) if trm_list else pd.DataFrame()
	df_ml = pd.DataFrame(ml_list) if ml_list else pd.DataFrame()
	
	if df_trm.empty:
		raise ValueError("TRM list should not be empty")
	
	# Reset index for TRM
	df_trm = df_trm.reset_index(drop=True)
	
	# Standardize ML transaction reference column as this is coming like this in menalaos programme
	if "Transaction_Reference" in df_ml.columns:
		df_ml = df_ml.rename(columns={"Transaction_Reference": "transaction_reference"})
	
	# Convert Model_Confidence to float if it's a string  changes to avoid conflicts from input
	if "Model_Confidence" in df_ml.columns:
		df_ml["Model_Confidence"] = df_ml["Model_Confidence"].astype(float)
	
	# Merge TRM and ML data on transaction_reference   already made sure that it should be small letters
	merged_df = pd.merge(df_trm, df_ml, on="transaction_reference", how="left")
	
	decisions = []
	
	for _, row in merged_df.iterrows():
		
		# TRM dictionary
		trm_dict = {col: row[col] for col in df_trm.columns}
		
		# ML dictionary (if ML data exists)
		ml_dict = {}
		if "Model_Prediction" in row and not pd.isna(row["Model_Prediction"]):
			ml_dict = {
				"Model_Prediction": int(row.get("Model_Prediction")),
				"Model_Confidence": float(row.get("Model_Confidence", 0)),
				"Transaction_Details_categorical": row.get("Transaction_Details_categorical", {}),
				"Transaction_Details_Numerical": row.get("Transaction_Details_Numerical", {})
			}
		else:
			print(f"Warning: No ML data found for transaction {row['transaction_reference']}. Using TRM decision only.")
		
		# TRM alerts
		all_alerts = trm_dict.get("all_alerts", [])
		
		# Make final decision
		try:
			decision = make_final_decision(trm_dict, ml_dict)
			
			# Ensure Reason dictionary exists
			if "Reason" not in decision:
				decision["Reason"] = {}
			
			# Add TRM alerts
			decision["Reason"]["reason_of_trm_engine"] = all_alerts
			
			decisions.append(decision)
		
		except Exception as e:
			print(f"Error processing transaction {row['transaction_reference']}: {e}")
			decisions.append({
				"transaction_reference": row["transaction_reference"],
				"scores": {},
				"reason_of_trm_engine": all_alerts
			})
	
	# Convert decisions to DataFrame and then to dict
	df_final = pd.DataFrame(decisions)
	return df_final.to_dict(orient="records"), decisions


def convert_numpy_types(d):
	"""Recursively convert NumPy types to native Python types for JSON."""
	if isinstance(d, dict):
		return {k: convert_numpy_types(v) for k, v in d.items()}
	elif isinstance(d, list):
		return [convert_numpy_types(x) for x in d]
	elif isinstance(d, np.generic):  # covers int64, float64, bool_
		return d.item()
	else:
		return d


# ------------------ Example Usage ------------------
if __name__ == "__main__":
	trm_list = [
		{
			"transaction_reference": "123",
			"rule_engine_decision": "ACKNOWLEDGE",
			"totalActiveRules": 2,
			"all_alerts": [
				"Alert triggered for C6262BD5-707E-40B1-92AB-0C81FDE232A0"
			],
			"fired_rules": [
				{
					"rule_id": 1,
					"system_identifier": "C6262BD5-707E-40B1-92AB-0C81FDE232A0",
					"risk_value_map": "medium_risk_rules",
					"rule_name": "Rule 1",
					"alert_display_text": "Alert triggered for C6262BD5-707E-40B1-92AB-0C81FDE232A0",
					"alert_computer_text": "FIELD_C6262BD5 = @ACTUAL = @EXPECTED|FIELD_C6262BD5 = @ACTUAL > @EXPECTED|FIELD_C6262BD5 = @ACTUAL > @EXPECTED",
					"alert_computer_numeric": "FIELD_C6262BD5 = 442.78 = 442.78|FIELD_C6262BD5 = 833.61 > 360.16|FIELD_C6262BD5 = 705.06 > 449.76"
				}
			],
			"trm_weights": [],
			"decision_weights": []
		},
		{
			"transaction_reference": "192983665",
			"rule_engine_decision": "ACKNOWLEDGE",
			"totalActiveRules": 5,
			"all_alerts": [
				"Alert triggered for AA0A9D86-608F-48CE-AB70-3833781884BB",
				"Alert triggered for 03A3DD2A-8C6B-445C-A687-DD412011E3B2",
				"Alert triggered for 8B98A25D-8A29-42C9-B7F6-5855839BCA6F",
				"Alert triggered for 08F3F100-42C5-4BFF-9DDF-AB93612D60CF"
			],
			"fired_rules": [
				{
					"rule_id": 1,
					"system_identifier": "AA0A9D86-608F-48CE-AB70-3833781884BB",
					"risk_value_map": "low_medium_risk_rules",
					"rule_name": "Rule 1",
					"alert_display_text": "Alert triggered for AA0A9D86-608F-48CE-AB70-3833781884BB",
					"alert_computer_text": "FIELD_AA0A9D86 = @ACTUAL >= @EXPECTED|FIELD_AA0A9D86 = @ACTUAL < @EXPECTED|FIELD_AA0A9D86 = @ACTUAL <= @EXPECTED|FIELD_AA0A9D86 = @ACTUAL >= @EXPECTED",
					"alert_computer_numeric": "FIELD_AA0A9D86 = 1156.23 >= 913.96|FIELD_AA0A9D86 = 1713.94 < 2599.6400000000003|FIELD_AA0A9D86 = 1785.5 <= 2127.86|FIELD_AA0A9D86 = 2302.7200000000003 >= 1325.01"
				},
				{
					"rule_id": 2,
					"system_identifier": "03A3DD2A-8C6B-445C-A687-DD412011E3B2",
					"risk_value_map": "medium_risk_rules",
					"rule_name": "Rule 2",
					"alert_display_text": "Alert triggered for 03A3DD2A-8C6B-445C-A687-DD412011E3B2",
					"alert_computer_text": "FIELD_03A3DD2A = @ACTUAL < @EXPECTED|FIELD_03A3DD2A = @ACTUAL != @EXPECTED|FIELD_03A3DD2A = @ACTUAL > @EXPECTED",
					"alert_computer_numeric": "FIELD_03A3DD2A = 1975.55 < 2589.4|FIELD_03A3DD2A = 2613.03 != 2649.078328014712|FIELD_03A3DD2A = 2259.62 > 1305.89"
				},
				{
					"rule_id": 3,
					"system_identifier": "8B98A25D-8A29-42C9-B7F6-5855839BCA6F",
					"risk_value_map": "high_risk_rules",
					"rule_name": "Rule 3",
					"alert_display_text": "Alert triggered for 8B98A25D-8A29-42C9-B7F6-5855839BCA6F",
					"alert_computer_text": "FIELD_8B98A25D = 2676.04|FIELD_8B98A25D = 1524.11|FIELD_8B98A25D = 968.05",
					"alert_computer_numeric": "FIELD_8B98A25D = 2676.04|FIELD_8B98A25D = 1524.11|FIELD_8B98A25D = 968.05"
				},
				{
					"rule_id": 4,
					"system_identifier": "08F3F100-42C5-4BFF-9DDF-AB93612D60CF",
					"risk_value_map": "high_risk_rules",
					"rule_name": "Rule 4",
					"alert_display_text": "Alert triggered for 08F3F100-42C5-4BFF-9DDF-AB93612D60CF",
					"alert_computer_text": "FIELD_08F3F100 = 894.22",
					"alert_computer_numeric": "FIELD_08F3F100 = 894.22"
				}
			],
			"trm_weights": [],
			"decision_weights": [
				{
					"flag": 0,
					"trm_weight": 0.44,
					"ml_weight": 0.41,
					"given_threshold": 0.63
				}
			]
		},
		{
			"transaction_reference": "690782902",
			"rule_engine_decision": "INDECISIVE",
			"totalActiveRules": 3,
			"all_alerts": [
				"Alert triggered for CFE82F13-F712-45A6-9F37-92A633DAA93D",
				"Alert triggered for B24F7D3D-7285-4BF8-B880-D813A45A967A"
			],
			"fired_rules": [
				{
					"rule_id": 1,
					"system_identifier": "CFE82F13-F712-45A6-9F37-92A633DAA93D",
					"risk_value_map": "low_risk_rules",
					"rule_name": "Rule 1",
					"alert_display_text": "Alert triggered for CFE82F13-F712-45A6-9F37-92A633DAA93D",
					"alert_computer_text": "FIELD_CFE82F13 = @ACTUAL <= @EXPECTED",
					"alert_computer_numeric": "FIELD_CFE82F13 = 1868.86 <= 2072.48"
				},
				{
					"rule_id": 2,
					"system_identifier": "B24F7D3D-7285-4BF8-B880-D813A45A967A",
					"risk_value_map": "low_medium_risk_rules",
					"rule_name": "Rule 2",
					"alert_display_text": "Alert triggered for B24F7D3D-7285-4BF8-B880-D813A45A967A",
					"alert_computer_text": "FIELD_B24F7D3D = @ACTUAL = @EXPECTED",
					"alert_computer_numeric": "FIELD_B24F7D3D = 218.03 = 218.03"
				}
			],
			"trm_weights": [],
			"decision_weights": []
		}
	]
	
	# ml_list = [{'Model_Confidence': '0.5167351', 'Model_Prediction': '1', 'Transaction_Details_Numerical': {'Amount_converted': 32415.45, 'hour': 16}, 'Transaction_Details_categorical': {'From_Bank_Country': 'Kerala', 'From_Bank_Name': 'Thiruvananthapuram Branch_Bank', 'To_Bank_Country': 'Thiruvananthapuram, Kerala'}, 'Transaction_Reference': "123"}]
	ml_list = [
		{
			'Model_Confidence': '0.5167351',
			'Model_Prediction': '1',
			'Transaction_Details_Numerical': {
				'Amount_converted': 32415.45,
				'hour': 16
			},
			'Transaction_Details_categorical': {
				'From_Bank_Country': 'Kerala',
				'From_Bank_Name': 'Thiruvananthapuram Branch_Bank',
				'To_Bank_Country': 'Thiruvananthapuram, Kerala'
			},
			'Transaction_Reference': "123"
		},
		{
			'Model_Confidence': '0.8254312',
			'Model_Prediction': '1',
			'Transaction_Details_Numerical': {
				'Amount_converted': 48622.47,
				'hour': 14
			},
			'Transaction_Details_categorical': {
				'From_Bank_Country': 'Bihar',
				'From_Bank_Name': 'Patna Branch_Bank',
				'To_Bank_Country': 'Delhi, India'
			},
			'Transaction_Reference': "192983665"
		},
		{
			'Model_Confidence': '0.9134572',
			'Model_Prediction': '1',
			'Transaction_Details_Numerical': {
				'Amount_converted': 49876.55,
				'hour': 18
			},
			'Transaction_Details_categorical': {
				'From_Bank_Country': 'Tamil Nadu',
				'From_Bank_Name': 'Chennai Branch_Bank',
				'To_Bank_Country': 'Tamil Nadu, India'
			},
			'Transaction_Reference': "690782902"
		}
	]
	
	decisions_df, decisions_list = run_trm_ml_engine(trm_list, ml_list)  ### this is the main function
	step2_end = time.time()
	# Step 3 — Convert NumPy types (for API serialization)
	step3_start = time.time()
	decisions_list = convert_numpy_types(decisions_list)  #### include this helper function for correct output
	step3_end = time.time()
	# Step 4 — Prepare JSON Payload
	step4_start = time.time()
	api_payload = json.dumps(decisions_list, indent=4)
	step4_end = time.time()
	# --- Optionally print payload ---
	print(api_payload)
