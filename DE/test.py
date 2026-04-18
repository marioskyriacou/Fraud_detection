import decision_engine_version as de
import json

if __name__ == "__main__":
	trm_list = [{'all_alerts': ['Alert triggered for BAA6B436-AA90-47BE-B7A7-188850B8FE7F', 'Alert triggered for 03A07D2C-AAD2-4AB2-AC7D-A9E64071E19E',
	                            'Alert triggered for 36BF0B42-5114-4202-B250-5FDE29FCCDCA', 'Alert triggered for 6BCBAAD0-7DB8-4208-866C-E83A6BAD4C3D'],
	             'decision_weights': [], 'fired_rules': [{
			'alert_computer_numeric': 'FIELD_BAA6B436 = 2979.04|FIELD_BAA6B436 = 2365.25 != 2451.094698080698|FIELD_BAA6B436 = 765.41|FIELD_BAA6B436 = 1479.41 = 1479.41',
			'alert_computer_text': 'FIELD_BAA6B436 = 2979.04|FIELD_BAA6B436 = @ACTUAL != @EXPECTED|FIELD_BAA6B436 = 765.41|FIELD_BAA6B436 = @ACTUAL = @EXPECTED',
			'alert_display_text': 'Alert triggered for BAA6B436-AA90-47BE-B7A7-188850B8FE7F',
			'risk_value_map': 'low_medium_risk_rules', 'rule_id': 1, 'rule_name': 'Rule 1',
			'system_identifier': 'BAA6B436-AA90-47BE-B7A7-188850B8FE7F'},
			{'alert_computer_numeric': 'FIELD_03A07D2C = 2859.37 < 3649.45',
			 'alert_computer_text': 'FIELD_03A07D2C = @ACTUAL < @EXPECTED',
			 'alert_display_text': 'Alert triggered for 03A07D2C-AAD2-4AB2-AC7D-A9E64071E19E',
			 'risk_value_map': 'low_risk_rules', 'rule_id': 2, 'rule_name': 'Rule 2',
			 'system_identifier': '03A07D2C-AAD2-4AB2-AC7D-A9E64071E19E'}, {
				'alert_computer_numeric': 'FIELD_36BF0B42 = 2897.72 < 3681.41|FIELD_36BF0B42 = 2127.86 >= 1144.74',
				'alert_computer_text': 'FIELD_36BF0B42 = @ACTUAL < @EXPECTED|FIELD_36BF0B42 = @ACTUAL >= @EXPECTED',
				'alert_display_text': 'Alert triggered for 36BF0B42-5114-4202-B250-5FDE29FCCDCA',
				'risk_value_map': 'low_medium_risk_rules', 'rule_id': 3, 'rule_name': 'Rule 3',
				'system_identifier': '36BF0B42-5114-4202-B250-5FDE29FCCDCA'}, {
				'alert_computer_numeric': 'FIELD_6BCBAAD0 = 525.1|FIELD_6BCBAAD0 = 1700.54 <= 2233.23|FIELD_6BCBAAD0 = 2101.11 >= 1778.0|FIELD_6BCBAAD0 = 1801.35 <= 2533.94',
				'alert_computer_text': 'FIELD_6BCBAAD0 = 525.1|FIELD_6BCBAAD0 = @ACTUAL <= @EXPECTED|FIELD_6BCBAAD0 = @ACTUAL >= @EXPECTED|FIELD_6BCBAAD0 = @ACTUAL <= @EXPECTED',
				'alert_display_text': 'Alert triggered for 6BCBAAD0-7DB8-4208-866C-E83A6BAD4C3D',
				'risk_value_map': 'high_risk_rules', 'rule_id': 4, 'rule_name': 'Rule 4',
				'system_identifier': '6BCBAAD0-7DB8-4208-866C-E83A6BAD4C3D'}],
	             'rule_engine_decision': 'INDECISIVE', 'totalActiveRules': 4, 'transaction_reference': '4fa3208f-9e23-42dc-b330-844829d0c12c',
	             'trm_weights': []},
	            {'all_alerts': ['Alert triggered for BAA6B436-AA90-47BE-B7A7-188850B8FE7F', 'Alert triggered for 03A07D2C-AAD2-4AB2-AC7D-A9E64071E19E',
	                            'Alert triggered for 36BF0B42-5114-4202-B250-5FDE29FCCDCA', 'Alert triggered for 6BCBAAD0-7DB8-4208-866C-E83A6BAD4C3D'],
	             'decision_weights': [], 'fired_rules': [{
		            'alert_computer_numeric': 'FIELD_BAA6B436 = 2979.04|FIELD_BAA6B436 = 2365.25 != 2451.094698080698|FIELD_BAA6B436 = 765.41|FIELD_BAA6B436 = 1479.41 = 1479.41',
		            'alert_computer_text': 'FIELD_BAA6B436 = 2979.04|FIELD_BAA6B436 = @ACTUAL != @EXPECTED|FIELD_BAA6B436 = 765.41|FIELD_BAA6B436 = @ACTUAL = @EXPECTED',
		            'alert_display_text': 'Alert triggered for BAA6B436-AA90-47BE-B7A7-188850B8FE7F',
		            'risk_value_map': 'low_medium_risk_rules', 'rule_id': 1, 'rule_name': 'Rule 1',
		            'system_identifier': 'BAA6B436-AA90-47BE-B7A7-188850B8FE7F'},
		            {'alert_computer_numeric': 'FIELD_03A07D2C = 2859.37 < 3649.45',
		             'alert_computer_text': 'FIELD_03A07D2C = @ACTUAL < @EXPECTED',
		             'alert_display_text': 'Alert triggered for 03A07D2C-AAD2-4AB2-AC7D-A9E64071E19E',
		             'risk_value_map': 'low_risk_rules', 'rule_id': 2, 'rule_name': 'Rule 2',
		             'system_identifier': '03A07D2C-AAD2-4AB2-AC7D-A9E64071E19E'}, {
			            'alert_computer_numeric': 'FIELD_36BF0B42 = 2897.72 < 3681.41|FIELD_36BF0B42 = 2127.86 >= 1144.74',
			            'alert_computer_text': 'FIELD_36BF0B42 = @ACTUAL < @EXPECTED|FIELD_36BF0B42 = @ACTUAL >= @EXPECTED',
			            'alert_display_text': 'Alert triggered for 36BF0B42-5114-4202-B250-5FDE29FCCDCA',
			            'risk_value_map': 'low_medium_risk_rules', 'rule_id': 3, 'rule_name': 'Rule 3',
			            'system_identifier': '36BF0B42-5114-4202-B250-5FDE29FCCDCA'}, {
			            'alert_computer_numeric': 'FIELD_6BCBAAD0 = 525.1|FIELD_6BCBAAD0 = 1700.54 <= 2233.23|FIELD_6BCBAAD0 = 2101.11 >= 1778.0|FIELD_6BCBAAD0 = 1801.35 <= 2533.94',
			            'alert_computer_text': 'FIELD_6BCBAAD0 = 525.1|FIELD_6BCBAAD0 = @ACTUAL <= @EXPECTED|FIELD_6BCBAAD0 = @ACTUAL >= @EXPECTED|FIELD_6BCBAAD0 = @ACTUAL <= @EXPECTED',
			            'alert_display_text': 'Alert triggered for 6BCBAAD0-7DB8-4208-866C-E83A6BAD4C3D',
			            'risk_value_map': 'high_risk_rules', 'rule_id': 4, 'rule_name': 'Rule 4',
			            'system_identifier': '6BCBAAD0-7DB8-4208-866C-E83A6BAD4C3D'}],
	             'rule_engine_decision': 'INDECISIVE', 'totalActiveRules': 4, 'transaction_reference': 'asdasd',
	             'trm_weights': []}]
	
	ml_list = [{'Model_Confidence': '0.5167351', 'Model_Prediction': '1', 'Transaction_Details_Numerical': {'Amount_converted': 32415.45, 'hour': 16},
	            'Transaction_Details_categorical': {'From_Bank_Country': 'Kerala', 'From_Bank_Name': 'Thiruvananthapuram Branch_Bank',
	                                                'To_Bank_Country': 'Thiruvananthapuram, Kerala'},
	            'Transaction_Reference': '4fa3208f-9e23-42dc-b330-844829d0c12c'},
	           {'Model_Confidence': '0.5167351', 'Model_Prediction': '1', 'Transaction_Details_Numerical': {'Amount_converted': 32415.45, 'hour': 16},
	            'Transaction_Details_categorical': {'From_Bank_Country': 'Kerala', 'From_Bank_Name': 'Thiruvananthapuram Branch_Bank',
	                                                'To_Bank_Country': 'Thiruvananthapuram, Kerala'},
	            'Transaction_Reference': 'asdasd'}]
	decisions_df, decisions_list = de.run_trm_ml_engine(trm_list, ml_list)  ### this is the main function
	# Step 3 — Convert NumPy types (for API serialization)
	decisions_list = de.convert_numpy_types(decisions_list)  #### include this helper function for correct output
	# Step 4 — Prepare JSON Payload
	api_payload = json.dumps(decisions_list, indent=4)
	# --- Optionally print payload ---
	print(api_payload)
