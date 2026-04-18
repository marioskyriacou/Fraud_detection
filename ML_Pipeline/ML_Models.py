from Libraries import pd, np, Union, List, train_test_split, tqdm, Counter, dump, logging
from Libraries import NearMiss, SMOTE
from Libraries import (classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
                       roc_auc_score, precision_recall_curve, auc, roc_curve, geometric_mean_score,
                       balanced_accuracy_score)
from Libraries import sns, plt
from Libraries import DecisionTreeClassifier, RandomForestClassifier, XGBClassifier, LGBMClassifier, BaseEstimator
from Libraries import StratifiedKFold, GridSearchCV

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger_ml = logging.getLogger('Machine Learning')


# SPLIT + RESAMPLING DATA -------------------------------------------------------------
def split_data(data: pd.DataFrame, target_col: str = 'Is_Fraud', selected_features: list = None, test_size: float = 0.1,
               val_size: float = 0.2):
	# Potential errors ------------------------------------------------------------------------------
	if not isinstance(data, pd.DataFrame):
		logger_ml.error(f'Input must be a pandas DataFrame, received type: {type(data)}.')
		raise TypeError('Input must be a pandas DataFrame.')
	if data.empty:
		logger_ml.error("Input DataFrame 'data' cannot be empty.")
		raise ValueError("Input DataFrame 'data' cannot be empty.")
	if target_col not in data.columns:
		logger_ml.error(f"Target column '{target_col}' not found in DataFrame columns.")
		raise ValueError(f"Target column '{target_col}' not found.")
	
	if not (0.0 <= test_size <= 1.0):
		logger_ml.error(f"Test threshold must be between 0.0 and 1.0. Received: {test_size}")
		raise ValueError("Test threshold must be between 0.0 and 1.0.")
	
	if not (0.0 <= val_size <= 1.0):
		logger_ml.error(f"Validation threshold must be between 0.0 and 1.0. Received: {val_size}")
		raise ValueError("Validation threshold must be between 0.0 and 1.0.")
	# Split Data  ------------------------------------------------------------------------------
	try:
		X, y = data.drop(columns=[target_col]), data[target_col]
		if X.shape[1] == 0:
			logger_ml.error("Input DataFrame 'data' cannot be empty.")
			raise TypeError("Input DataFrame 'data' cannot be empty.")
		if y.nunique() < 2:
			logger_ml.error(f"Target column '{target_col}' has only one unique value. Cannot stratify.")
			raise ValueError("Target must have at least two unique values for stratification.")
	
	except Exception as e:
		logger_ml.error(f"Error during data preparation for splitting features & target: {e}", exc_info=True)
		raise RuntimeError(f"Failed to prepare data: {e}")
	logger_ml.info(f'Features data shape: {X.shape}. Target counts: {Counter(y)}')
	
	# Select features ------------------------------------------------------------------------------
	if selected_features:
		try:
			X = X[selected_features]
			logger_ml.info(f'>>>Data:{X.shape} Features Selected: {selected_features}')
		except Exception as e:
			raise RuntimeError(f"Error during data preparation for Data Selection: {e}")
	
	try:
		X_train_original, X_val_test, y_train_original, y_val_test = train_test_split(X, y,
		                                                                              test_size=test_size + val_size,
		                                                                              stratify=y)
		logger_ml.debug(f"Original Train shape: {X_train_original.shape}-->{Counter(y_train_original)}")
		logger_ml.debug(f"Original Val/Test shape: {X_val_test.shape}-->{Counter(y_val_test)}")
	except Exception as e:
		logger_ml.error(f"Error during data preparation for training and testing: {e}", exc_info=True)
		raise RuntimeError(f"Error during data preparation for training and testing: {e}")
	
	# Split train into train + validation
	# nm = NearMiss(version=1, sampling_strategy=0.5)  # keep most minority
	# X_train, y_train = nm.fit_resample(X_train, y_train)
	# print(f'>>> NearMiss: {Counter(y_train)}')
	try:
		sm = SMOTE(k_neighbors=3, sampling_strategy=0.5, random_state=42)
		X_train, y_train = sm.fit_resample(X_train_original, y_train_original)
		logger_ml.info(f'SMOTE applied. | Resampled train counts: {X_train.shape}-->{Counter(y_train)} ')
	except Exception as e:
		X_train, y_train = X_train_original, y_train_original
		logger_ml.warning("SMOTE failed. Proceeding with original, imbalanced training data.")
	try:
		# test_split_ratio = test_size / test_size + val_size
		X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=test_size,
		                                                stratify=y_val_test)
		logger_ml.info(f"Split Val/Test into Validation and Test.")
	except Exception as e:
		logger_ml.error(f"Error during final validation/test split: {e}", exc_info=True)
		raise RuntimeError(f"Failed during validation/test split: {e}")
	
	logger_ml.info("-" * 40)
	logger_ml.info(f'X_train shape: {X_train.shape} | y_train counts: {Counter(y_train)}')
	logger_ml.info(f'X_val shape: {X_val.shape} | y_val counts: {Counter(y_val)}')
	logger_ml.info(f'X_test shape: {X_test.shape} | y_test counts: {Counter(y_test)}')
	logger_ml.info("-" * 40)
	
	return X_train, X_val, X_test, y_train, y_val, y_test


# EVALUATION METRICS -----------------------------------------------------------------------------------------------------
def evaluate_basic(y_true: Union[pd.Series, np.ndarray, List[Union[int, str]]],
                   y_pred: Union[pd.Series, np.ndarray, List[Union[int, str]]],
                   plot_cm: bool = False):
	# Ensure Type ------------------------------------------------------------------------
	if not isinstance(y_true, (pd.Series, np.ndarray, list)) or not isinstance(y_pred, (pd.Series, np.ndarray, list)):
		logger_ml.error("y_true and y_pred must be array-like (Series, ndarray, or list).")
		raise TypeError("y_true and y_pred must be array-like.")
	# Ensure Len ------------------------------------------------------------------------
	if len(y_true) != len(y_pred):
		logger_ml.error(f"Input arrays must have the same length. True: {len(y_true)}, Pred: {len(y_pred)}")
		raise ValueError("y_true and y_pred must have the same length.")
	
	try:
		acc = accuracy_score(y_true, y_pred)
		bal_acc = balanced_accuracy_score(y_true, y_pred)
		f1 = f1_score(y_true, y_pred)
		prec = precision_score(y_true, y_pred)
		recall = recall_score(y_true, y_pred)
		gmean = geometric_mean_score(y_true, y_pred)
		cm = confusion_matrix(y_true, y_pred)
	except Exception as e:
		logger_ml.critical(f"Error during metric calculation. Check input data labels: {e}", exc_info=True)
		raise RuntimeError(f"Metric calculation failed: {e}")
	
	# logger_ml.info("-" * 30)
	# logger_ml.info("Classification Report:")
	# report = classification_report(y_true, y_pred, zero_division=0)
	# for line in report.splitlines():
	# 	logger_ml.info(line)
	
	logger_ml.info("-" * 30)
	logger_ml.info(
		f"Accuracy: {acc:.4f} -- Balanced Accuracy: {bal_acc:.4f} --  F1 Score: {f1:.4f} -- Precision: {prec:.4f} -- Recall: {recall:.4f} -- G-Mean: {gmean:.4f}")
	
	if plot_cm:
		try:
			sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
			plt.xlabel("Predicted")
			plt.ylabel("Actual")
			plt.title("Confusion Matrix")
			plt.show()
		except Exception as e:
			logger_ml.error(f"Error during confusion matrix plotting: {e}", exc_info=True)
			logger_ml.info(f"Confusion Matrix (Numpy Array):\n{cm}")
	else:
		logger_ml.info(f'Confusion Matrix (Numpy Array):\n{cm}')
	
	return acc, bal_acc, f1, prec, recall, gmean


def evaluate_advanced(y_true: Union[pd.Series, np.ndarray, List[int]],
                      anomaly_scores: Union[pd.Series, np.ndarray, List[float]],
                      top_n: int = None,
                      plot_curves: bool = False):
	try:
		y_true = np.array(y_true, dtype=np.int32)
		anomaly_scores = np.array(anomaly_scores, dtype=np.float64)
		
		if len(y_true) != len(anomaly_scores):
			logger_ml.error(
				f"Input arrays must have the same length. True: {len(y_true)}, Scores: {len(anomaly_scores)}")
			raise ValueError("y_true and anomaly_scores must have the same length.")
		
		if y_true.sum() == 0 or (len(y_true) - y_true.sum()) == 0:
			logger_ml.error("y_true must contain at least one positive and one negative sample for AUC metrics.")
			raise ValueError("Target array must contain both classes.")
	
	except Exception as e:
		logger_ml.error(f"Input validation/conversion error: {e}", exc_info=True)
		raise TypeError(f"Invalid input data: {e}")
	
	try:
		roc_auc = roc_auc_score(y_true, anomaly_scores)
		precision, recall, _ = precision_recall_curve(y_true, anomaly_scores)
		pr_auc = auc(recall, precision)
		# Precision @ N
		if top_n is None:
			top_n = int(y_true.sum())  # true anomalies
		idx = np.argsort(anomaly_scores)[::-1]  # descending order
		p_at_n = y_true[idx][:top_n].sum() / top_n
	
	except Exception as e:
		logger_ml.critical(f"Error during metric calculation: {e}", exc_info=True)
		raise RuntimeError(f"Metric calculation failed: {e}")
	
	logger_ml.info("-" * 30)
	logger_ml.info(f"ROC AUC: {roc_auc:.4f}")
	logger_ml.info(f"PR AUC: {pr_auc:.4f}")
	logger_ml.info(f"Precision @ {top_n}: {p_at_n:.4f}")
	logger_ml.info("-" * 30)
	
	if plot_curves:
		try:
			# ROC Curve
			fpr, tpr, _ = roc_curve(y_true, anomaly_scores)
			plt.figure(figsize=(12, 5))
			plt.subplot(1, 2, 1)
			plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
			plt.plot([0, 1], [0, 1], '--', color='gray')
			plt.xlabel("False Positive Rate")
			plt.ylabel("True Positive Rate")
			plt.title("ROC Curve")
			plt.legend()
			
			# PR Curve
			plt.subplot(1, 2, 2)
			plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
			plt.xlabel("Recall")
			plt.ylabel("Precision")
			plt.title("Precision-Recall Curve")
			plt.legend()
			
			plt.tight_layout()
			plt.show()
		except Exception as e:
			logger_ml.error(f"Error during curve plotting: {e}", exc_info=True)


# SELECT OPTIMAL ML MODEL -----------------------------------------------------------------------------------------------
def optimal_ML_Model(supervised_models: dict,
                     X_train: pd.DataFrame,
                     y_train: pd.Series,
                     n_splits: int = 3):
	logger_ml.info(f'>>>>> MACHINE LEARNING MODEL COMPARISON ({n_splits}-FOLD CV) <<<<<<<')
	# Error prevention -----------------------------------------------------------
	if not supervised_models:
		logger_ml.warning("No supervised models provided. Returning None.")
		return None
	if not isinstance(X_train, pd.DataFrame) or not isinstance(y_train, pd.Series):
		logger_ml.error("X_train must be DataFrame and y_train must be Series.")
		raise TypeError("X_train must be DataFrame and y_train must be Series.")
	
	if X_train.empty or y_train.empty:
		logger_ml.error("Training data cannot be empty.")
		raise ValueError("Training data cannot be empty.")
	if not supervised_models:
		logger_ml.warning("No supervised models provided. Returning None.")
		return None, None, {}
	if n_splits < 2:
		logger_ml.error("n_splits must be greater than 1 for cross-validation.")
		raise ValueError("n_splits must be greater than 1.")
	
	# Variables --------------------------------------------------------------
	best_model, best_name = None, None
	best_f1, best_prec, best_recall, best_bal_acc = -1, -1, -1, -1
	model_details = {}
	X_train = X_train.reset_index(drop=True)
	y_train = y_train.reset_index(drop=True)
	try:
		skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
	except Exception as e:
		logger_ml.critical(f"Error initializing StratifiedKFold: {e}", exc_info=True)
		raise RuntimeError("Failed to initialize StratifiedKFold.")
	
	# Cross Validation + Model Training -------------------------------- ---
	for name, model in supervised_models.items():
		logger_ml.info(f"\n{'=' * 20} CV for Model: {name}  {'=' * 20}")
		logger_ml.info(f'Training data shape: {y_train.shape}')
		cv_acc, cv_f1, cv_prec, cv_recall, cv_bal_acc, cv_gmean = [], [], [], [], [], []
		try:
			for i, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
				logger_ml.debug(f'>>>>> Starting Fold {i + 1}/{n_splits} ')
				X_tr, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
				y_tr, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
				
				model.fit(X_tr, y_tr)
				y_val_pred = model.predict(X_val_fold)
				
				acc, bal_acc, f1, prec, recall, g_mean = evaluate_basic(y_true=y_val_fold, y_pred=y_val_pred,
				                                                        plot_cm=False)
				cv_acc.append(acc)
				cv_f1.append(f1)
				cv_prec.append(prec)
				cv_recall.append(recall)
				cv_bal_acc.append(bal_acc)
				cv_gmean.append(g_mean)
		except Exception as e:
			logger_ml.error(f"Error during CV for model {name}: {e}", exc_info=True)
			continue
		
		# Average CV metrics
		avg_f1 = np.mean(cv_f1)
		avg_prec = np.mean(cv_prec)
		avg_recall = np.mean(cv_recall)
		avg_bal_acc = np.mean(cv_bal_acc)
		avg_acc = np.mean(cv_acc)
		avg_gmean = np.mean(cv_gmean)
		logger_ml.info(f"{'-' * 80}")
		logger_ml.info(f"CV Metrics — F1: {avg_f1:.4f}, Precision: {avg_prec:.4f}, Recall: {avg_recall:.4f}, "
		               f"Balanced Acc: {avg_bal_acc:.4f} , Accuracy: {avg_acc:.4f}, G MEan: {avg_gmean:.4f}")
		logger_ml.info(f"{'-' * 80}")
		
		# Store results
		model_details[name] = {
			'model': model,
			'b_acc': avg_bal_acc,
			'f1': avg_f1,
			'precision': avg_prec,
			'recall': avg_recall,
			'g_mean': avg_gmean,
			'accuracy': avg_acc}
		
		# Best Model Selection ------------------------------------------------------------------------
		if avg_f1 > best_f1 and avg_recall > best_recall and avg_bal_acc > best_bal_acc and avg_prec > best_prec:
			best_model = model
			best_f1 = avg_f1
			best_recall = avg_recall
			best_bal_acc = avg_bal_acc
			best_prec = avg_prec
			best_name = name
		if best_model is None:
			logger_ml.warning("No models successfully trained or evaluated. Returning None.")
			return None, None, model_details
	
	logger_ml.info(f"\n{'=' * 25} BEST MODEL SUMMARY {'=' * 25}")
	logger_ml.info(f"Best Model : {best_model.__class__.__name__}")
	logger_ml.info(
		f"F1: {best_f1:.4f} | Recall: {best_recall:.4f} |  Balanced Accuracy: {best_bal_acc:.4f} | Precision: {best_prec:.4f} ")
	logger_ml.info(f'{'=' * 50}')
	
	return best_model, best_name, model_details


# HYPERPARAMETER TUNING -------------------------------------------------------------------------------------------------
def hyperparameter_tuning(empty_model: BaseEstimator, X_grid: pd.DataFrame, y_grid: pd.Series):
	def get_model_and_params(model_instance: BaseEstimator, y: pd.Series):
		if isinstance(model_instance, DecisionTreeClassifier):
			# dt_param_dict = {
			#     'criterion': ['gini', 'entropy', 'log_loss'],  'max_depth': [None, 10, 50, 100], 'min_samples_split': [2, 5, 10, 20],
			#     'min_samples_leaf': [1, 2, 5, 10],'ccp_alpha': [0.0, 0.001, 0.01, 0.1],'class_weight': [{0:0.3, 1:0.7}, 'balanced']
			#                 }
			dt_param_dict = {'criterion': ['gini', 'entropy'], 'max_depth': [5, 30, 50, 100],
			                 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2, 5],
			                 'ccp_alpha': [0.01, 0.1, 1.0], 'class_weight': [{0: 0.3, 1: 0.7},
			                                                                 {0: 0.7, 1: 0.3}, 'balanced']}
			
			return DecisionTreeClassifier(), dt_param_dict
		
		elif isinstance(model_instance, RandomForestClassifier):
			# rf_param_dict = {
			#     'n_estimators': [30, 80, 150, 500], 'criterion': ['gini', 'entropy', 'log_loss'], 'max_depth': [None, 10, 20, 30],
			#     'min_samples_split': [2, 5, 10, 20], 'min_samples_leaf': [1, 2, 4, 8], 'bootstrap': [True, False],
			#     'class_weight': [{0:0.3, 1:0.7}, 'balanced'], ccp_alpha': [0.0, 0.001, 0.01, 0.1]
			#                 }
			rf_param_dict = {'n_estimators': [30, 100], 'criterion': ['gini', 'entropy'],
			                 'max_depth': [10, 50], 'min_samples_split': [2, 10],
			                 'min_samples_leaf': [1, 2], 'bootstrap': [True, False],
			                 'class_weight': [{0: 0.3, 1: 0.7}, 'balanced'], 'ccp_alpha': [0.0, 0.01]}
			return RandomForestClassifier(), rf_param_dict
		
		elif isinstance(model_instance, XGBClassifier):
			# xgb_param_dict = {
			#     'n_estimators': [30, 50, 80, 150], 'learning_rate': [0.01, 0.1, 1], 'max_depth': [5, 10, 20],
			#     'min_child_weight': [1, 5, 10], 'subsample': [0.2, 0.5, 1.0], 'gamma': [0, 0.5, 1, 10],
			#     'reg_alpha': [0, 0.1, 1, 10], 'reg_lambda': [0.1, 1, 10], 'scale_pos_weight': [1, 5, 10, 20]
			#                 }
			ratio = (len(y) - sum(y)) / sum(y)
			xgb_param_dict = {
				'n_estimators': [30, 80, ], 'learning_rate': [0.1, 1], 'max_depth': [5, 20],
				'min_child_weight': [1, 5], 'subsample': [0.2, 1.0], 'gamma': [0, 1, 10],
				'reg_alpha': [1, 10], 'reg_lambda': [1, 10], 'scale_pos_weight': [1, ratio]
			}
			return XGBClassifier(), xgb_param_dict
		
		elif isinstance(model_instance, LGBMClassifier):
			# lgb_param_dict = {
			#     'n_estimators': [30, 50, 80, 150], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [5, 10, 20],
			#     'min_child_samples': [1, 5, 10], 'bagging_fraction': [0.2, 0.5, 1.0], 'min_split_gain': [0, 0.5, 1, 10],
			#     'reg_alpha': [0, 0.1, 1, 10], 'reg_lambda': [0.1, 1, 10], 'scale_pos_weight': [1, 5, 10, 20]
			#                 }
			ratio = (len(y) - sum(y)) / sum(y)
			lgb_param_dict = {
				'n_estimators': [30, 80, ], 'learning_rate': [0.1, 1], 'max_depth': [5, 20],
				'min_child_weight': [1, 5], 'subsample': [0.2, 1.0], 'min_child_samples': [1, 10],
				'reg_alpha': [1, 10], 'reg_lambda': [1, 10], 'scale_pos_weight': [1, ratio]
			}
			return LGBMClassifier(), lgb_param_dict
		
		else:
			logger_ml.error(f"Unsupported model type: {model_instance.__class__.__name__}")
			raise ValueError(f"Model: Unsupported estimator type: {model_instance.__class__.__name__}")
	
	if not isinstance(X_grid, pd.DataFrame) or not isinstance(y_grid, pd.Series):
		logger_ml.error("X_grid must be DataFrame and y_grid must be Series.")
		raise TypeError("X_grid must be DataFrame and y_grid must be Series.")
	if X_grid.empty or y_grid.empty:
		logger_ml.error("Grid search input data cannot be empty.")
		raise ValueError("Input data cannot be empty.")
	
	scoring = {
		'f1_macro': 'f1_macro',
		'f1_weighted': 'f1_weighted',
		'recall_macro': 'recall_macro',
		'average_precision': 'average_precision',
		'balanced_accuracy': 'balanced_accuracy'}
	# Define model parameters to tune
	try:
		
		model_instance, param_dict = get_model_and_params(model_instance=empty_model, y=y_grid)
		model_name = model_instance.__class__.__name__
		logger_ml.info(f'Estimator: {model_name}. Parameters to search: {len(param_dict)}')
	except Exception as e:
		logger_ml.critical(f"Error during model/parameter initialization: {e}", exc_info=True)
		raise RuntimeError("Failed to initialize model or parameters.")
	try:
		grid_search = GridSearchCV(
			estimator=model_instance, param_grid=param_dict,
			scoring=scoring, refit='f1_macro', cv=5, verbose=1)
		logger_ml.info(f"Starting Grid Search. Refit metric: 'f1_macro', CV folds: 5.")
		
		# Fit grid Search ------------------------------------
		grid_search.fit(X_grid, y_grid)
		best_estimator = grid_search.best_estimator_
		best_parmas = grid_search.best_params_
		best_score = grid_search.best_score_
	except Exception as e:
		logger_ml.critical(f"Critical error during GridSearchCV fit for {model_name}: {e}", exc_info=True)
		raise RuntimeError(f"GridSearchCV fit failed: {e}")
	
	logger_ml.info('-' * 100)
	logger_ml.info(f"Best Estimator Found: {best_estimator}")
	logger_ml.info(f"Best Parameters: {best_parmas}")
	logger_ml.info(f"Best Score (f1_macro): {best_score:.4f}")
	logger_ml.info('-' * 100)
	
	return best_estimator, best_parmas


# SAVE BEST MODEL  -----------------------------------------------------------------------------------------------------------
def save_model(name: str, model: BaseEstimator, X: pd.DataFrame, file_path: str = "shared/best_model_details.joblib") -> bool:
	# --- Input Validation ---
	if not isinstance(model, BaseEstimator):
		logger_ml.error("Input 'model' must be an sklearn-compatible estimator.")
		raise TypeError("Input 'model' must be an sklearn-compatible estimator.")
	if not isinstance(X, pd.DataFrame) or X.empty:
		logger_ml.error("Input 'X' must be a non-empty pandas DataFrame.")
		raise ValueError("Input 'X' must be a non-empty pandas DataFrame.")
	if not hasattr(model, 'predict'):
		logger_ml.error("The model instance does not appear to be fitted (missing 'predict' method).")
		raise RuntimeError("Model must be fitted before saving.")
	
	sur_model = None
	
	if name != 'DecisionTreeClassifier':
		logger_ml.info(f"Creating Decision Tree surrogate model for {name}...")
		try:
			y_surr = model.predict(X)
			sur_model = DecisionTreeClassifier(max_depth=10, random_state=42).fit(X, y_surr)
		except Exception as e:
			logger_ml.error(f"Error creating surrogate model for {name}. Saving primary model only. Error: {e}",
			                exc_info=True)
			sur_model = None
	else:
		logger_ml.info("Model is a Decision Tree; surrogate model creation skipped.")
	model_data = {
		"name": name,
		"model": model,
		'surmodel': sur_model,
		"features_names": list(X.columns)
	}
	try:
		dump(model_data, file_path)
		logger_ml.info(f"Model details saved successfully to: {file_path}")
		return True
	except Exception as e:
		logger_ml.critical(f"CRITICAL: Failed to save model to {file_path}. Error: {e}", exc_info=True)
		return False


# EVALUATE FINAL MODEL ON UNSEEN DATA ----------------------------------------------------------------------------------
def evaluate_final_model(model: BaseEstimator, X_unseen: pd.DataFrame,
                         y_true: Union[pd.Series, np.ndarray, List[int]]) -> None:
	if not isinstance(model, BaseEstimator):
		logger_ml.error("Input 'model' must be an sklearn-compatible estimator.")
		raise TypeError("Input 'model' must be an sklearn-compatible estimator.")
	if not isinstance(X_unseen, pd.DataFrame) or X_unseen.empty:
		logger_ml.error("Input 'X_unseen' must be a non-empty pandas DataFrame.")
		raise ValueError("Input 'X_unseen' must be a non-empty pandas DataFrame.")
	if len(X_unseen) != len(y_true):
		logger_ml.error("X_unseen and y_true must have the same length.")
		raise ValueError("X_unseen and y_true must have the same length.")
	anomaly_scores = None
	try:
		if hasattr(model, "predict_proba"):
			anomaly_scores = model.predict_proba(X_unseen)[:, 1]
			logger_ml.info(f"Using 'predict_proba' (Class 1 probability) for scores.")
		elif hasattr(model, "decision_function"):
			anomaly_scores = model.decision_function(X_unseen)
			logger_ml.info(f"Using 'decision_function' for scores.")
		elif hasattr(model, "score_samples"):
			anomaly_scores = -model.score_samples(X_unseen)  # Invert for anomaly likelihood
			logger_ml.info(f"Using inverted 'score_samples' for scores.")
	except Exception as e:
		logger_ml.error(f"Error generating anomaly scores for {model.__class__.__name__}: {e}", exc_info=True)
		pass
	
	if anomaly_scores is not None and model is not None:
		logger_ml.info(f"\n{'=' * 20} EVALUATION ON TEST SET ({X_unseen.shape}) {'=' * 20}")
		try:
			y_pred_test = model.predict(X_unseen)
			_, _, _, _, _, _ = evaluate_basic(y_true=y_true, y_pred=y_pred_test, plot_cm=False)
			logger_ml.info("Basic classification evaluation complete.")
			evaluate_advanced(y_true=y_true, anomaly_scores=anomaly_scores, top_n=None, plot_curves=False)
		except Exception as e:
			logger_ml.error(f"Error during basic evaluation (hard predictions): {e}", exc_info=True)
			logger_ml.warning("Basic evaluation skipped due to failure.")
	else:
		logger_ml.error(
			f'No suitable score generation method found or score generation failed for {model.__class__.__name__}.')
		raise ValueError(f'No anomaly scores available for {model.__class__.__name__}')
