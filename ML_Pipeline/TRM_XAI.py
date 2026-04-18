"""
Author:		Menelaos Artemiou
Created: 	20/10/2025
Version: 	3.0
"""

# Imports
import time
import os
import numpy as np
import json
from sklearn.tree import _tree
from joblib import load


def sanitize_for_json(obj):
	"""
	Recursively traverse obj and replace non-serializable values with strings.
	"""
	if isinstance(obj, dict):
		return {k: sanitize_for_json(v) for k, v in obj.items()}
	elif isinstance(obj, list):
		return [sanitize_for_json(v) for v in obj]
	else:
		try:
			json.dumps(obj)  # Test if serializable
			return obj
		except TypeError:
			return str(obj)


# Explain one tree


def explain_with_tree(tree_model, sample, feature_names, target_names=None):
	tree = tree_model.tree_
	node = 0
	explanation = []
	
	while tree.feature[node] != _tree.TREE_UNDEFINED:
		feat = tree.feature[node]
		thresh = tree.threshold[node]
		val = sample[0, feat]
		
		if val <= thresh:
			explanation.append(f"{feature_names[feat]} = {val:.2f} <= {thresh:.2f}")
			node = tree.children_left[node]
		else:
			explanation.append(f"{feature_names[feat]} = {val:.2f} > {thresh:.2f}")
			node = tree.children_right[node]
	
	# leaf probabilities (for classifiers)
	if hasattr(tree, "value") and target_names is not None:
		class_counts = tree.value[node][0]
		class_probs = class_counts / class_counts.sum()
		pred_idx = np.argmax(class_probs)
		confidence = class_probs[pred_idx]
		pred_class = target_names[pred_idx]
	else:
		pred_class = None
		confidence = None
	
	return explanation, pred_class, confidence


def explainability(data_file: str, preprocessing_file: str, processed_input: np.array, original_input: np.array) -> None:
	# Record the start time
	start_time_explainability = time.time()
	
	# Load the saved dictionary
	data = data_file
	preprocessing = preprocessing_file
	model_name = data['name']  # one of: "DecisionTree", "RandomForest", "XGBoost", "LightGBM"
	model = data['model']
	feature_names = data['features_names']
	target_names = data["model"].classes_
	
	result = []
	for i, value in enumerate(processed_input["processed_data"]):
		model_prediction = ""
		model_confidence = ""
		reference = processed_input['bankTransactionID'][i]
		print(i, reference, processed_input["processed_data"][i])
		sample = np.array(list(processed_input["processed_data"][i].values()))
		sample = sample.reshape(1, -1)
		if model_name == "DecisionTreeClassifier":
			explanation, pred_class, conf = explain_with_tree(model, sample, feature_names, target_names)
			logger.info(f"Model: Decision Tree")
			# logger.info("\n".join(" - " + s for s in explanation))
			logger.info(f"Prediction: {pred_class}, Confidence: {conf:.2f}")
			model_prediction = pred_class
			model_confidence = conf
		elif model_name == "RandomForestClassifier":
			surrogate = data['surmodel']
			explanation, pred_class, conf = explain_with_tree(surrogate, sample, feature_names, target_names)
			probs = model.predict_proba(sample)[0]
			logger.info(f"Model: Random Forest (with Surrogate Explanation)")
			# logger.info("\n".join(" - " + s for s in explanation))
			logger.info(f"Prediction: {target_names[np.argmax(probs)]}, Confidence: {probs.max():.2f}")
			model_prediction = target_names[np.argmax(probs)]
			model_confidence = probs.max()
		elif model_name == "XGBoostClassifier":
			surrogate = data['surmodel']
			explanation, pred_class, conf = explain_with_tree(surrogate, sample, feature_names, target_names)
			probs = model.predict_proba(sample)[0]
			logger.info(f"Model: XGBoost")
			# logger.info("\n".join(" - " + s for s in explanation))
			logger.info(f"Prediction: {target_names[np.argmax(probs)]}, Confidence: {probs.max():.2f}")
			model_prediction = target_names[np.argmax(probs)]
			model_confidence = probs.max()
		elif model_name == "LightGBMClassifier":
			surrogate = data['surmodel']
			explanation, pred_class, conf = explain_with_tree(surrogate, sample, feature_names, target_names)
			probs = model.predict_proba(sample)[0]
			logger.info(f"Model: LightGBM")
			# logger.info("\n".join(" - " + s for s in explanation))
			logger.info(f"Prediction: {target_names[np.argmax(probs)]}, Confidence: {probs.max():.2f}")
			model_prediction = target_names[np.argmax(probs)]
			model_confidence = probs.max()
		else:
			raise ValueError("Invalid model_name. Choose from: DecisionTree, RandomForest, XGBoost, LightGBM")
		
		cat_col = preprocessing['hc'] + preprocessing['lc']
		# logger.info(cat_col)
		all_col = feature_names
		# logger.info(all_col)
		num_col = [item for item in all_col if item not in cat_col]
		# logger.info(num_col)
		num_col = list(set(num_col) & set(all_col))
		# logger.info(num_col)
		cat_col = list(set(cat_col) & set(all_col))
		# logger.info(cat_col)
		scaler = preprocessing['s']
		
		import re
		
		# Initialize dictionary to store the first value after '=' for each feature
		values_dict = {}
		
		# Regular expression to match conditions and extract the value after '='
		pattern = r"([^=]+)=\s*([0-9.]+)"
		
		# Process each condition and extract the value after '='
		for condition in explanation:
			match = re.match(pattern, condition)
			if match:
				field, value = match.groups()
				field = field.strip()
				value = float(value)
				
				# Store the first value encountered for each field (if not already present)
				if field not in values_dict:
					values_dict[field] = value
		
		# Output the result: dictionary with key and single value for each feature
		"""for field, value in values_dict.items():
			logger.info(f"{field}: {value}")"""
		
		filtered_cat_columns = [key for key in cat_col if key in values_dict]
		filtered_num_columns = [key for key in num_col if key in values_dict]
		# Final result dictionaries to store matched and unmatched values
		cat_dict = {key: original_input[i][key] for key in filtered_cat_columns if key in original_input[i]}
		num_dict = {key: original_input[i][key] for key in filtered_num_columns if key in original_input[i]}
		# Iterate through the keys of the values_dict
		# for key, value_from_values_dict in values_dict.items():
		# 	if key in preprocessing["lc"]:  # Check if the key exists in other_dict
		# 		sub_dict = preprocessing['rlem'][key]  # Get the sub-dictionary for that key
		#
		# 		# If the value exists in the sub_dict, add it to the result dictionary
		# 		if value_from_values_dict in sub_dict:
		# 			cat_dict[key] = sub_dict[value_from_values_dict]
		# 		else:
		# 			# Add to num_dict with key name and value
		# 			num_dict[key] = value_from_values_dict
		# 	elif key in preprocessing["hc"]:  # Check if the key exists in other_dict
		# 		sub_dict = preprocessing['rvfem'][key]  # Get the sub-dictionary for that key
		#
		# 		# If the value exists in the sub_dict, add it to the result dictionary
		# 		if value_from_values_dict in sub_dict:
		# 			cat_dict[key] = sub_dict[value_from_values_dict]
		# 		else:
		# 			# Add to num_dict with key name and value
		# 			num_dict[key] = value_from_values_dict
		# 	else:
		# 		# If the key doesn't exist in other_dict, add to num_dict with key name and value
		# 		num_dict[key] = value_from_values_dict
		
		# Output the matched and num_dict dictionaries
		"""logger.info("Matched Results:")
		logger.info(cat_dict)
	
		logger.info("\nUnmatched Results:")
		logger.info(num_dict)
	
		logger.info(scaler.feature_names_in_)"""
		
		# Create the new array that will store the results
		result_array = []
		
		"""filled_indexes = []  # List to store indexes filled with num_dict values
		
		for index, name in enumerate(scaler.feature_names_in_):
			# If the name is in the num_dict, add its value, otherwise add negative infinity for debug reasons
			if name in num_dict:
				result_array.append(num_dict[name])
				filled_indexes.append(index)  # Record the index where the dict value was added
			else:
				result_array.append(np.nextafter(-np.inf, 0, dtype=np.float32))
		
		# logger.info("Indexes filled with values from num_dict:", filled_indexes)
		
		# Convert the result list to a numpy array
		result_array = np.array(result_array)
		
		# Output the result
		# logger.info(result_array)
		
		final = scaler.inverse_transform([result_array])
		# logger.info(float(final[0][0]))
		
		keys = scaler.feature_names_in_
		values = final.ravel()
		# Create a dictionary with the keys and values from the specified indexes
		num_result_dict = {keys[i]: values[i] for i in filled_indexes}"""
		
		# logger.info(num_result_dict)
		
		if model_prediction != "":
			logger.info(f"Model Prediction: {model_prediction}")
		else:
			raise ValueError(f"Empty or Invalid Model prediction: {model_prediction}")
		
		if model_confidence != "":
			logger.info(f"Model Confidence: {float(model_confidence):.2f}")
		else:
			raise ValueError(f"Empty or Invalid Model Confidence: {float(model_confidence):.2f}")
		
		"""if cat_dict:
			logger.info(cat_dict)"""
		
		"""if num_result_dict:
			logger.info(num_result_dict)"""
		
		output = {"Transaction_Reference": reference,
		          "Model_Prediction": model_prediction,
		          "Model_Confidence": model_confidence,
		          "Transaction_Details_categorical": cat_dict,
		          "Transaction_Details_Numerical": num_dict
		          }
		
		result.append(output)
	# logger.info(output)
	
	result = sanitize_for_json(result)
	# After the loop, try to convert the list to a JSON string
	try:
		json_output = json.dumps(result, indent=4)
		logging.info("JSON conversion successful.")
	except (TypeError, OverflowError) as e:
		logging.warning(f"Error during JSON conversion: {e}")
		json_output = None
	
	# Print the resulting JSON string, if conversion was successful
	if json_output:
		logging.info(json_output)
	else:
		logging.error("No valid JSON output to print.")
	
	# Record the end time
	end_time_explainability = time.time()
	
	# Calculate the elapsed time
	elapsed_time_explainability = end_time_explainability - start_time_explainability
	# Calculate weeks, days, hours, minutes, seconds, and milliseconds
	weeks = int(elapsed_time_explainability // (7 * 24 * 3600))  # 7 days in a week
	elapsed_time_explainability %= (7 * 24 * 3600)  # Remaining seconds after extracting weeks
	
	days = int(elapsed_time_explainability // (24 * 3600))
	elapsed_time_explainability %= (24 * 3600)  # Remaining seconds after extracting days
	
	hours = int(elapsed_time_explainability // 3600)
	elapsed_time_explainability %= 3600  # Remaining seconds after extracting hours
	
	minutes = int(elapsed_time_explainability // 60)
	elapsed_time_explainability %= 60  # Remaining seconds after extracting minutes
	
	seconds = int(elapsed_time_explainability)
	milliseconds = int((elapsed_time_explainability - seconds) * 1000)
	
	# Print the result
	# print(f"Elapsed time: {weeks} weeks, {days} days, {hours} hours, {minutes} minutes, {seconds} seconds, {milliseconds} milliseconds")
	
	return json_output


import logging
import traceback

filename = os.path.splitext(os.path.basename(__file__))[0] + ".log"
logger = logging.getLogger(__name__)
logging.basicConfig(filename=filename, encoding='utf-8', level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%d-%m-%Y %H:%M:%S')

if __name__ == "__main__":
	start_time = time.time()
	
	# xai_report = explainability(model_dict, pre_preprocessing_dict, processed_data_dict, data_dict)
	
	end_time = time.time()
	
	# Calculate the elapsed time
	elapsed_time = end_time - start_time
	# Calculate weeks, days, hours, minutes, seconds, and milliseconds
	weeks = int(elapsed_time // (7 * 24 * 3600))  # 7 days in a week
	elapsed_time %= (7 * 24 * 3600)  # Remaining seconds after extracting weeks
	
	days = int(elapsed_time // (24 * 3600))
	elapsed_time %= (24 * 3600)  # Remaining seconds after extracting days
	
	hours = int(elapsed_time // 3600)
	elapsed_time %= 3600  # Remaining seconds after extracting hours
	
	minutes = int(elapsed_time // 60)
	elapsed_time %= 60  # Remaining seconds after extracting minutes
	
	seconds = int(elapsed_time)
	milliseconds = int((elapsed_time - seconds) * 1000)
	
	# Print the result
	print(f"Elapsed time: {weeks} weeks, {days} days, {hours} hours, {minutes} minutes, {seconds} seconds, {milliseconds} milliseconds")
