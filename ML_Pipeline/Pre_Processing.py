from Libraries import os, np, pd, Tuple, List, Union, tqdm, MinMaxScaler, LabelEncoder, json, logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger_preprocessing = logging.getLogger('Pre_processing')


# ------------------- Data Reading ------------------- #
def read_data_object(json_object: str) -> pd.DataFrame:
	"""Reads json file data into a pandas DataFrame."""
	try:
		json_object = json.loads(json_object)
		data = pd.DataFrame(json_object)
		logger_preprocessing.info(f'>>> Load Data :{type(json_object)} --> Convert {type(data)}')
		return data
	except json.JSONDecodeError as e:
		raise ValueError(f"Input string is not valid JSON format: {e}")
	except ValueError as e:
		raise ValueError(f"JSON structure could not be converted to a simple DataFrame. Details: {e}")


# ------------------- Store Columns ------------------- #
def store_column(data: pd.DataFrame, col_name: Union[str, List[str]]) -> Tuple[
	pd.DataFrame, Union[pd.Series, pd.DataFrame]]:
	# check if dataframe is empty
	if data.empty:
		logger_preprocessing.error("Input DataFrame 'data' cannot be empty.")
		raise ValueError("Input DataFrame 'data' cannot be empty.")
	# check if columns are in the dataframe
	cols_to_drop = [col_name] if isinstance(col_name, str) else col_name
	for col in cols_to_drop:
		if col not in data.columns:
			logger_preprocessing.error(f"Column '{col}' not found in the DataFrame.")
			raise KeyError(f"Column '{col}' not found in the DataFrame.")
	logger_preprocessing.info(f'>>> Dropped Columns  {cols_to_drop}')
	
	dropped_col = data[col_name].copy()
	data = data.drop(columns=col_name)
	return data, dropped_col


# ------------------- Null Columns Removal ------------------- #
def remove_null_cols(data: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
	"""Removes columns with null percentage above the given threshold."""
	if data.empty:
		logger_preprocessing.error("Input DataFrame 'data' cannot be empty.")
		raise ValueError("Input DataFrame 'data' cannot be empty.")
	
	logger_preprocessing.info(f'>>>>> Removing null columns (threshold={threshold}) <<<<<')
	null_ratios = data.isnull().mean()
	removed_cols = null_ratios[null_ratios >= threshold].index.tolist()
	data = data.drop(columns=removed_cols)
	logger_preprocessing.info(f'... Removed columns: {removed_cols} --- Data: {data.shape}')
	return data


# ------------------- Null Row Filling ------------------- #
def fill_null_rows(data: pd.DataFrame) -> pd.DataFrame:
	"""Fills missing values: categorical with mode, numeric with median."""
	logger_preprocessing.info(f'>>>>> Start null Values  <<<<<')
	
	if not isinstance(data, pd.DataFrame):
		logger_preprocessing.error('Input must be DataFrame.')
		raise TypeError('Input must be DataFrame.')
	
	if data.empty:
		logger_preprocessing.error("Input DataFrame 'data' cannot be empty.")
		raise ValueError("Input DataFrame 'data' cannot be empty.")
	
	null_count = data.isnull().sum().sum()
	if null_count == 0:
		logger_preprocessing.info(f"No missing values: {null_count}")
		return data
	
	# Data Type selection -----------------
	logger_preprocessing.info(f"Initial total missing values: {null_count}")
	try:
		cat_cols = data.select_dtypes(include=['object', 'bool']).columns
		numeric_cols = data.select_dtypes(include=['number']).columns
	except Exception as e:
		logger_preprocessing.error(f"Error selecting column dtypes: {e}", exc_info=True)
		raise ValueError("Error selecting column dtypes.")
	
	try:
		for cc in tqdm.tqdm(cat_cols, desc='Fill categorical cols'):
			if not data[cc].mode().empty:
				data[cc] = data[cc].fillna(data[cc].mode()[0])
			else:
				data[cc] = data[cc].fillna("Unknown")
	except Exception as e:
		logger_preprocessing.error(f"Error filling categorical col {cc}:{e}", exc_info=True)
	
	try:
		for nc in tqdm.tqdm(numeric_cols, desc='Fill numeric cols'):
			median_val = data[nc].median()
			if pd.isna(median_val):
				logger_preprocessing.warning(
					f"Column '{nc}' median calculation failed (NaN). Skipping fill for this column.")
				continue
			data[nc] = data[nc].fillna(data[nc].median())
	except Exception as e:
		logger_preprocessing.error(f"Critical error during numeric column filling on column {nc}: {e}", exc_info=True)
	
	remaining_null_count = data.isnull().sum().sum()
	logger_preprocessing.info(f"Process complete. Initial nulls: {null_count}, Remaining nulls: {remaining_null_count}")
	logger_preprocessing.info('>>>>> Null row filling process finished <<<<<')
	return data


# ------------------- Duplicate Removal ------------------- #
def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
	"""Removes duplicate rows."""
	logger_preprocessing.info(f'>>>>> Removing duplicate rows <<<<<')
	if not isinstance(data, pd.DataFrame):
		logger_preprocessing.error('Input must be DataFrame.')
		raise TypeError('Input must be DataFrame.')
	if data.empty:
		logger_preprocessing.error("Input DataFrame 'data' cannot be empty.")
		raise ValueError("Input DataFrame 'data' cannot be empty.")
	try:
		clean_data = data.drop_duplicates().reset_index(drop=True)
		logger_preprocessing.info(f'... Removed {len(data) - len(clean_data)} duplicates')
		return clean_data
	except Exception as e:
		logger_preprocessing.critical(f"A critical error occurred during duplicate removal or index reset: {e}",
		                              exc_info=True)
		raise RuntimeError(f"Duplicate removal failed: {e}")


# ------------------- High Cardinality Columns ------------------- #
def remove_high_cardinality_cols(data: pd.DataFrame, cardinality_thresh: float = 0.9) -> Tuple[list, pd.DataFrame]:
	"""Removes columns with very high cardinality."""
	if not isinstance(data, pd.DataFrame):
		logger_preprocessing.error('Input must be DataFrame.')
		raise TypeError('Input must be DataFrame.')
	
	if data.empty:
		logger_preprocessing.error("Input DataFrame 'data' cannot be empty.")
		raise ValueError("Input DataFrame 'data' cannot be empty.")
	
	if not (0.0 <= cardinality_thresh <= 1.0):
		logger_preprocessing.error(f"Cardinality threshold must be greater than 0.")
		raise ValueError(f"Cardinality threshold must be greater than 0.")
	logger_preprocessing.info(f'>>>>> Removing high cardinality columns (threshold={cardinality_thresh}) <<<<<')
	
	very_high_card_cols = [c for c in data.columns if data[c].nunique() / len(data) >= cardinality_thresh]
	if not very_high_card_cols:
		logger_preprocessing.info('No columns met the high cardinality removal threshold.')
		return very_high_card_cols, data
	
	try:
		clean_data = data.drop(columns=very_high_card_cols)
	except Exception as e:
		raise RuntimeError(f"Failed during high cardinality column removal: {e}")
	
	logger_preprocessing.info(f'... Removed columns: {very_high_card_cols}')
	logger_preprocessing.info(f'... Data shape after removal: {clean_data.shape}')
	return very_high_card_cols, clean_data


# ------------------- Normalization & Encoding ------------------- #
def normalize_encode(data: pd.DataFrame, cardinality_thresh: float = 0.2, target_col: str = 'Is_Fraud'):
	"""
	Encodes categorical columns and scales numeric columns.
	Uses Label Encoding for low-cardinality and Frequency Encoding for high-cardinality.
	"""
	if not isinstance(data, pd.DataFrame):
		logger_preprocessing.error(f'Input must be a pandas DataFrame, received type: {type(data)}.')
		raise TypeError('Input must be a pandas DataFrame.')
	
	if data.empty:
		logger_preprocessing.error("Input DataFrame 'data' cannot be empty.")
		raise ValueError("Input DataFrame 'data' cannot be empty.")
	
	if target_col not in data.columns:
		logger_preprocessing.error(f"Target column '{target_col}' not found in DataFrame columns.")
		raise ValueError(f"Target column '{target_col}' not found.")
	
	if not (0.0 <= cardinality_thresh <= 1.0):
		logger_preprocessing.error(f"Cardinality threshold must be between 0.0 and 1.0. Received: {cardinality_thresh}")
		raise ValueError("Cardinality threshold must be between 0.0 and 1.0.")
	
	# SPLIT DATA ------------------------------------------------------------------------------
	logger_preprocessing.info(f'>>>>> Encoding and Normalization <<<<<')
	X, y = data.drop(columns=target_col), data[target_col]
	# Variables --------------------------------------------------------------------------------
	label_encoder_mappings, reverse_label_encoder_mappings = {}, {}
	freq_encoder_mappings, reverse_freq_encoder_mappings = {}, {}
	scaler = MinMaxScaler()
	low_card_cols, high_card_cols = [], []
	# columns selection  ------------------------------------------------------------------------------
	try:
		cat_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()
		num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
		
		if len(cat_cols) == 0 or len(num_cols) == 0:
			logger_preprocessing.error('No Numeric or Categorical columns found in DataFrame.')
			raise ValueError('No Numeric or Categorical columns found in DataFrame.')
		
		if np.sum([len(cat_cols), len(num_cols)]) != X.shape[1]:
			logger_preprocessing.error('No Numeric columns found in DataFrame.')
			raise ValueError(
				f'Sum of Categorical and numerical {np.sum(len(cat_cols), len(num_cols))}not equal to {X.shape[1]}.')
	
	except Exception as e:
		logger_preprocessing.error(f"Failed during numeric columns removal: {e}")
		raise ValueError(f"Failed to identify column types: {e}")
	
	# Ratio selection  ------------------------------------------------------------------------------
	main_sum_unique = sum([len(np.unique(X[col])) for col in cat_cols])
	if main_sum_unique == 0:
		logger_preprocessing.error(
			"All categorical columns are empty or entirely composed of missing values (NaN). Cannot calculate cardinality ratio.")
		raise ZeroDivisionError('Total unique categorical values is zero.')
	
	# Encoding categorical columns ------------------------------------------------------------------------------
	if cat_cols:
		logger_preprocessing.info("Starting categorical encoding...")
		for c in tqdm.tqdm(cat_cols, desc='Encoding categorical columns'):
			try:
				num_unique = X[c].nunique()
				ratio = num_unique / main_sum_unique
				if ratio >= cardinality_thresh:
					logger_preprocessing.info(f'Frequency Encoding (High Cardinality) -> {c} (Ratio: {ratio:.4f})')
					high_card_cols.append(c)
					freq_map = X[c].value_counts(normalize=True).to_dict()
					X[c] = X[c].map(freq_map)
					freq_encoder_mappings[c] = freq_map
					reverse_freq_encoder_mappings[c] = {v: k for k, v in freq_map.items()}
				else:
					logger_preprocessing.debug(f'Label Encoding (Low Cardinality) -> {c} (Ratio: {ratio:.4f})')
					le = LabelEncoder()
					low_card_cols.append(c)
					X[c] = le.fit_transform(X[c].astype(str))
					label_encoder_mappings[c] = dict(zip(le.classes_, range(len(le.classes_))))
					reverse_label_encoder_mappings[c] = {int(v): k for k, v in label_encoder_mappings[c].items()}
			except Exception as e:
				logger_preprocessing.error(f"Failed during label encoding of {c}: {e}")
	else:
		logger_preprocessing.info("No categorical columns found for encoding.")
	
	# Numeric Scaling columns ------------------------------------------------------------------------------
	if num_cols:
		logger_preprocessing.info("Starting Numeric Scaling...")
		try:
			# X_normal = X[y == 0].copy()
			scaler.fit(X[num_cols])
			X[num_cols] = scaler.transform(X[num_cols])
			logger_preprocessing.info(f'MinMax scaling applied to numeric columns: {num_cols}')
		except Exception as e:
			logger_preprocessing.error(f"Error during numeric scaling: {e}", exc_info=True)
			raise RuntimeError(f"Critical error during numeric scaling: {e}")
	else:
		logger_preprocessing.info("No categorical columns found for encoding.")
	
	# Numeric Scaling columns ------------------------------------------------------------------------------
	try:
		processed_data = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
		processed_data = processed_data.sample(frac=1, random_state=42).reset_index(drop=True)
	except Exception as e:
		logger_preprocessing.critical(f"Error during final data assembly or shuffling: {e}", exc_info=True)
		raise RuntimeError(f"Failed to assemble final processed data: {e}")
	
	logger_preprocessing.info(f'>>> Preprocessed data shape: {processed_data.shape}')
	return (processed_data, label_encoder_mappings, reverse_label_encoder_mappings,
	        freq_encoder_mappings, reverse_freq_encoder_mappings, low_card_cols, high_card_cols, scaler)


# ------------------- Export Data ------------------- #
def export_data(data: pd.DataFrame, path: str, name: str, format: str = 'csv') -> bool:
	if not isinstance(data, pd.DataFrame):
		logger_preprocessing.error(f'Input data must be a pandas DataFrame, received type: {type(data)}.')
		raise TypeError('Input data must be a pandas DataFrame.')
	
	if not isinstance(path, str) or not isinstance(name, str):
		logger_preprocessing.error(
			f'Path and name must be strings. Received path type: {type(path)}, name type: {type(name)}.')
		return False
	# Check Format ---------------------------------
	valid_formats = ['csv', 'json']
	if format.lower() not in valid_formats:
		logger_preprocessing.error(f"Unsupported format '{format}'. Must be one of: {valid_formats}.")
		return False
	#  Data Export -------------------------------------------------------------------
	full_path = os.path.join(path, f"{name}.{format.lower()}")
	try:
		if format.lower() == 'csv':
			data.to_csv(full_path, index=False)
			logger_preprocessing.info(f"DataFrame successfully exported as CSV to: {full_path}")
		elif format.lower() == 'json':
			data.to_json(full_path, orient='records', indent=4)
			logger_preprocessing.info(f"DataFrame successfully exported as JSON to: {full_path}")
		logger_preprocessing.info('>>>>> Data export finished <<<<<')
		return True
	
	except Exception as e:
		logger_preprocessing.critical(f"Unexpected error during directory creation: {e}", exc_info=True)
		return False

# ------------------- Reverse Encoding & MinMax Scaler------------------- #
# def reverse_label_encoding(encoded_value, column, reverse_label_encoder_mappings):
#     # Reverses label encoding for a given encoded value based on the column.
#     return reverse_label_encoder_mappings[column].get(encoded_value, None)
#
#
# def reverse_target_encoding(encoded_value, column, reverse_target_encoder_mappings):
#     # Reverse target encoding (approximate) by finding the closest match in the mapping.
#     if column in reverse_target_encoder_mappings:
#         mapping = reverse_target_encoder_mappings[column]
#         # Return the category with the closest mean target value
#         closest_category = min(mapping, key=lambda k: abs(mapping[k] - encoded_value))
#         return closest_category
#     return None
#
#
# def reverse_min_max_scaling(scaled_value, scaler, column):
#     # Reverses MinMax scaling for a given scaled value.
#     min_value, max_value = scaler.data_min_[scaler.feature_names_in_.tolist().index(column)], scaler.data_max_[
#         scaler.feature_names_in_.tolist().index(column)]
#     return (scaled_value * (max_value - min_value)) + min_value
