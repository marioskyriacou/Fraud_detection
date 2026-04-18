from Libraries import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger_fs = logging.getLogger('Feature Selection')


def anova_features(data: pd.DataFrame, target_col: str = 'Is_Fraud', threshold: float = 0.65, min_pct: float = 0.2) -> \
		List[str]:
	# Potential Errors ------------------------------------------------------------------------------------------------
	if not isinstance(data, pd.DataFrame):
		logger_fs.error(f'Input must be a pandas DataFrame, received type: {type(data)}.')
		raise TypeError('Input must be a pandas DataFrame.')
	
	if data.empty:
		logger_fs.error("Input DataFrame 'data' cannot be empty.")
		raise ValueError("Input DataFrame 'data' cannot be empty.")
	
	if target_col not in data.columns:
		logger_fs.error(f"Target column '{target_col}' not found in DataFrame columns.")
		raise ValueError(f"Target column '{target_col}' not found.")
	
	if not (0.0 <= threshold <= 1.0):
		logger_fs.error(f"ANOVA percentage threshold must be between 0.0 and 1.0. Received: {threshold}")
		raise ValueError("ANOVA percentage threshold must be between 0.0 and 1.0.")
	
	if not (0.0 <= min_pct <= 1.0):
		logger_fs.error(f"Minimum percentage threshold must be between 0.0 and 1.0. Received: {min_pct}")
		raise ValueError("Minimum percentage threshold must be between 0.0 and 1.0.")
	
	# Data Preparation  ---------------------------------------
	logger_fs.info(f'>>>>> ANOVA Feature Selection (Threshold: {threshold}, Min Pct: {min_pct}) <<<<<')
	try:
		numeric_data = data.select_dtypes(include=np.number).copy()
		if target_col not in numeric_data.columns:
			logger_fs.error(f"Target column '{target_col}' must be numeric for ANOVA (f_regression).")
			raise ValueError(f"Target column '{target_col}' must be numeric.")
		numeric_cols = numeric_data.columns.tolist()
		if len(numeric_cols) < 1:
			logger_fs.warning(
				f"No numeric features found besides the target column '{target_col}'. Returning empty list.")
			return []
		logger_fs.info(f'ANOVA columns--> {numeric_cols}')
		
		# Split features and target
		X, y = numeric_data.drop(columns=[target_col]), numeric_data[target_col]
	except Exception as e:
		logger_fs.error(f"Error during data preparation for ANOVA: {e}", exc_info=True)
		raise RuntimeError(f"Failed to prepare data: {e}")
	
	try:
		# ANOVA
		anova_selector = SelectKBest(score_func=f_regression, k='all')
		anova_selector.fit(X, y)
		anova_scores = pd.Series(anova_selector.scores_, index=X.columns)
		logger_fs.debug("ANOVA F-scores calculated successfully.")
		
		# Scale
		scaler = MinMaxScaler()
		scaled_scores = pd.Series(scaler.fit_transform(anova_scores.values.reshape(-1, 1)).flatten(),
		                          index=anova_scores.index)
		logger_fs.debug("F-scores scaled using MinMaxScaler.")
		
		# Selected features
		selected_features = scaled_scores[scaled_scores >= threshold].sort_values(ascending=False).index.tolist()
		logger_fs.info(f'Features selected by threshold ({threshold}): {len(selected_features)} features.')
		
		# ensure min features
		min_features = int(np.ceil(len(X.columns) * min_pct))
		if len(selected_features) < min_features:
			selected_features = scaled_scores.sort_values(ascending=False).index[:min_features].tolist()
			logger_fs.warning(
				f"Feature count ({len(selected_features)}) is below minimum ({min_features}). Selecting top {min_features} features.")
		
		logger_fs.info(f">>> Selected ANOVA Features (threshold={threshold}):\n{selected_features}")
	
	except Exception as e:
		logger_fs.critical(f"Critical error during ANOVA calculation or scaling: {e}", exc_info=True)
		raise RuntimeError(f"ANOVA feature selection failed: {e}")
	
	return selected_features


def mutual_information(data: pd.DataFrame, target_col: str = 'Is_Fraud', threshold: float = 0.01,
                       min_pct: float = 0.2) -> List[str]:
	# Potential Errors ------------------------------------------------------------------------------------------------
	if not isinstance(data, pd.DataFrame):
		logger_fs.error(f'Input must be a pandas DataFrame, received type: {type(data)}.')
		raise TypeError('Input must be a pandas DataFrame.')
	
	if data.empty:
		logger_fs.error("Input DataFrame 'data' cannot be empty.")
		raise ValueError("Input DataFrame 'data' cannot be empty.")
	
	if target_col not in data.columns:
		logger_fs.error(f"Target column '{target_col}' not found in DataFrame columns.")
		raise ValueError(f"Target column '{target_col}' not found.")
	
	if not (0.0 <= threshold <= 1.0):
		logger_fs.error(f"ANOVA percentage threshold must be between 0.0 and 1.0. Received: {threshold}")
		raise ValueError("ANOVA percentage threshold must be between 0.0 and 1.0.")
	
	if not (0.0 <= min_pct <= 1.0):
		logger_fs.error(f"Minimum percentage threshold must be between 0.0 and 1.0. Received: {min_pct}")
		raise ValueError("Minimum percentage threshold must be between 0.0 and 1.0.")
	
	logger_fs.info(f'>>>>> MUTUAL INFORMATION Feature Selection (Threshold: {threshold}, Min Pct: {min_pct}) <<<<<')
	try:
		X, y = data.drop(columns=[target_col]), data[target_col]
		if X.shape[1] == 0:
			logger_fs.error("Input DataFrame 'data' cannot be empty.")
			raise TypeError("Input DataFrame 'data' cannot be empty.")
	
	except Exception as e:
		logger_fs.error(f"Error during data preparation for Mutual Information: {e}", exc_info=True)
		raise RuntimeError(f"Failed to prepare data: {e}")
	
	# mutual information ----------
	try:
		mi = mutual_info_classif(X, y, random_state=42)
		mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
		logger_fs.info(f"... Calculate Mutual Information score")
		# Select features > threshold
		selected_features = mi_series[mi_series >= threshold].index.tolist()
		# minimum features based on percentage
		total_features = len(X.columns)
		min_features = int(np.ceil(total_features * min_pct))
		if len(selected_features) < min_features:
			selected_features = mi_series.index[:min_features].tolist()
			logger_fs.warning(
				f"Feature count ({len(selected_features)}) is below minimum ({min_features}). Selecting top {min_features} features based on rank.")
	
	except Exception as e:
		logger_fs.critical(f"Critical error during Mutual Information calculation: {e}", exc_info=True)
		raise RuntimeError(f"Mutual Information calculation failed. Check feature data types: {e}")
	
	logger_fs.info(
		f"Final selected MUTUAL INFORMATION Features (Total: {len(selected_features)}):\n{selected_features}")
	return selected_features


def xgb_features(data: pd.DataFrame, target_col: str = 'Is_Fraud', threshold: float = 0.01, min_pct: float = 0.2) -> \
		List[str]:
	# Potential Errors ------------------------------------------------------------------------------------------------
	if not isinstance(data, pd.DataFrame):
		logger_fs.error(f'Input must be a pandas DataFrame, received type: {type(data)}.')
		raise TypeError('Input must be a pandas DataFrame.')
	
	if data.empty:
		logger_fs.error("Input DataFrame 'data' cannot be empty.")
		raise ValueError("Input DataFrame 'data' cannot be empty.")
	
	if target_col not in data.columns:
		logger_fs.error(f"Target column '{target_col}' not found in DataFrame columns.")
		raise ValueError(f"Target column '{target_col}' not found.")
	
	if not (0.0 <= threshold <= 1.0):
		logger_fs.error(f"ANOVA percentage threshold must be between 0.0 and 1.0. Received: {threshold}")
		raise ValueError("ANOVA percentage threshold must be between 0.0 and 1.0.")
	
	if not (0.0 <= min_pct <= 1.0):
		logger_fs.error(f"Minimum percentage threshold must be between 0.0 and 1.0. Received: {min_pct}")
		raise ValueError("Minimum percentage threshold must be between 0.0 and 1.0.")
	
	logger_fs.info(f'>>>>> EMBEDDED Feature Selection (Threshold: {threshold}, Min Pct: {min_pct}) <<<<<')
	
	try:
		X, y = data.drop(columns=[target_col]), data[target_col]
		if X.shape[1] == 0:
			logger_fs.error("Input DataFrame 'data' cannot be empty.")
			raise TypeError("Input DataFrame 'data' cannot be empty.")
	except Exception as e:
		logger_fs.error(f"Error during data preparation for Mutual Information: {e}", exc_info=True)
		raise RuntimeError(f"Failed to prepare data: {e}")
	
	try:
		ratio = (len(y) - sum(y)) / sum(y)
		if pd.isnull(ratio):
			logger_fs.warning(f"Ratio of null count ({len(y)}) is NaN.")
			ratio = 0
	except Exception as e:
		logger_fs.error(f"Error during mutual information calculation: {e}", exc_info=True)
		raise RuntimeError(f"Failed to calculate mutual information: {e}")
	try:
		model = LGBMClassifier(n_estimators=50, learning_rate=0.1, scale_pos_weight=ratio, max_depth=20, verbosity=0)
		model.fit(X, y)
		logger_fs.info(f'..... Feature Selection -- Model train successful.')
		lgb_series = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
		lgb_series /= lgb_series.sum()
		logger_fs.debug(f"Normalized feature importance scores")
	except Exception as e:
		logger_fs.critical(f"Critical error during LGBM model training or importance extraction: {e}", exc_info=True)
		raise RuntimeError(f"LGBM model training failed: {e}")
	
	selected_features = lgb_series[lgb_series >= threshold].index.tolist()
	logger_fs.info(f'Features selected by normalized threshold ({threshold}): {len(selected_features)} features.')
	total_features = len(X.columns)
	min_features = int(np.ceil(total_features * min_pct))
	if len(selected_features) < min_features:
		selected_features = lgb_series.index[:min_features].tolist()
		logger_fs.warning(
			f"Feature count ({len(selected_features)}) is below minimum ({min_features}). Selecting top {min_features} features.")
	
	logger_fs.info(f"Final selected LGBM Features (Total: {len(selected_features)}):\n{selected_features}")
	return selected_features
