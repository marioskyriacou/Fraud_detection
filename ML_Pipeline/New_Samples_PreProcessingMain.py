from Libraries import load, pd, logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger_new_sample_pp = logging.getLogger('New Sample PreProcessing')


def preprocessing_new_samples(data: pd.DataFrame, processing_joblib_path: str, model_joblib_path: str):
	logger_new_sample_pp.info("Starting preprocessing of new samples.")
	logger_new_sample_pp.debug(f"Input data shape: {data.shape}")
	
	# VARIABLES ------------------------------------------------------------
	try:
		pre_preprocessing_dict = load(f'{processing_joblib_path}.joblib')
		model_dict = load(f'{model_joblib_path}.joblib')
		logger_new_sample_pp.info(f"Successfully loaded joblib files: {processing_joblib_path} and {model_joblib_path}")
	except FileNotFoundError as e:
		logger_new_sample_pp.error(f"Failed to load joblib files. Check file paths. Error: {e}")
		raise
	
	# load items
	lem, fem = pre_preprocessing_dict['lem'], pre_preprocessing_dict['fem']  # load encoders
	high_card_cols, low_card_cols = pre_preprocessing_dict['hc'], pre_preprocessing_dict['lc']  # load cardinality
	scaler_ = pre_preprocessing_dict['s']  # load scaler
	f_name = model_dict['features_names']  # Load Feature Model
	
	if not scaler_ or not f_name:
		logger_new_sample_pp.critical("Essential components (scaler or feature names) missing from joblib files.")
		raise KeyError("Essential components missing from joblib files (scaler or features_names).")
	
	# Store TransactionID ----------------------------------------------------------------------
	transaction_ids = data['bankTransactionID'].copy()  # Use .copy() to avoid SettingWithCopyWarning
	logger_new_sample_pp.debug(f"Stored {len(transaction_ids)} transaction IDs.")
	
	# Check missing values ------------------------------------------------
	missing_sum = data.isnull().sum().sum()
	if missing_sum > 0:
		missing_details = data.isnull().sum()[data.isnull().sum() > 0]
		logger_new_sample_pp.error(f"{missing_sum} total missing values found. Details:\n{missing_details.to_string()}")
		raise ValueError(f"{missing_sum} - Missing values found in the dataset. Data must be clean.")
	logger_new_sample_pp.info("No missing values found in the input data.")
	
	# Encoding ------------------------------------------------------------
	for c in data.columns:
		try:
			if c in lem and c in low_card_cols:
				# Map values in low cardinality -> label encoder
				data[c] = data[c].map(lem[c])
				if data[c].isnull().any():
					logger_new_sample_pp.warning(f"Column '{c}' (Low Card) encountered new category(s). Mapped to NaN.")
				logger_new_sample_pp.info(f"Applied Label Encoding map to column: {c}")
			
			elif c in fem and c in high_card_cols:
				# Map values in high cardinality -> frequency encoder
				data[c] = data[c].map(fem[c])
				if data[c].isnull().any():
					logger_new_sample_pp.warning(
						f"Column '{c}' (High Card) encountered new category(s). Mapped to NaN.")
				logger_new_sample_pp.info(f"Applied Frequency Encoding map to column: {c}")
		
		except TypeError as e:
			logger_new_sample_pp.error(f"Type error during encoding of column {c}. Data type mismatch? Error: {e}")
			raise
	
	# Normalize numeric columns ------------------------------------------
	
	if scaler_:
		cols_to_scale = [c for c in data.columns if c in scaler_.feature_names_in_]
		logger_new_sample_pp.info(f"Scaling {len(cols_to_scale)} numerical columns.")
		
		if not cols_to_scale:
			logger_new_sample_pp.warning("No columns found to scale based on scaler_.feature_names_in_.")
		try:
			data[cols_to_scale] = scaler_.transform(data[cols_to_scale])
		except ValueError as e:
			logger_new_sample_pp.error(f"Value Error during scaling. Check if columns are numeric. Error: {e}")
			raise
	else:
		logger_new_sample_pp.warning("Scaler object is missing; skipping normalization.")
	
	# Mapping Columns for the Model ----------------------------------------------------------------------
	
	data = data[f_name]
	missing_cols = set(f_name) - set(data.columns)
	extra_cols = set(data.columns) - set(f_name)  # Identify columns that are not needed
	
	if missing_cols:
		logger_new_sample_pp.error(f"Missing required model features in input data: {missing_cols}")
		raise ValueError(f"Missing columns in input data: {missing_cols}")
	
	logger_new_sample_pp.debug(f"Ignoring extra columns not required by the model: {extra_cols}")
	data = data[list(f_name)]
	logger_new_sample_pp.info(f"Final feature set selected. Data shape: {data.shape}")
	# Results -----------------------------------------------------
	json_processed_data, trd_id = [], []
	
	if not len(transaction_ids) == data.shape[0]:
		logger_new_sample_pp.error(
			f"Mismatch: IDs length ({len(transaction_ids)}) vs final data shape ({data.shape[0]}).")
		raise IndexError("Mismatch in transaction ID count and processed data row count.")
	
	json_processed_data = data.to_dict('records')
	for idx, tr_id in enumerate(transaction_ids):
		# row = data.iloc[idx].apply(lambda x: x).to_dict()
		# json_processed_data.append(row)
		trd_id.append(tr_id)
	
	data_dict = {'bankTransactionID': trd_id, 'processed_data': json_processed_data}
	
	if not len(json_processed_data) == data.shape[0]:
		logger_new_sample_pp.critical(
			f"Row count mismatch during final dictionary creation. {len(json_processed_data)} vs {data.shape[0]}")
		raise ValueError(f"{len(json_processed_data)} rows missing from input data")
	
	logger_new_sample_pp.info("Preprocessing completed successfully.")
	return data_dict, pre_preprocessing_dict, model_dict

# ----------------------------------------------------------------------------------------------------------
