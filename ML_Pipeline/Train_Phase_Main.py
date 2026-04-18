from Libraries import dump, pd, time, logging
from Libraries import DecisionTreeClassifier, RandomForestClassifier, XGBClassifier, LGBMClassifier

import Pre_Processing as pp
import Feature_Selection as fs
import ML_Models as mlmodels

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger_mlp = logging.getLogger('Machine Learning Pipeline')


# PRE PROCESSING PHASE MAIN ---------------------------------------------------------------------------------------------
def pre_processing_training(data: pd.DataFrame, target_col: str) -> pd.DataFrame:
	data, _ = pp.store_column(data=data, col_name=['Transaction_Date', 'bankTransactionID'])  #
	data = pp.remove_null_cols(data=data)
	data = pp.fill_null_rows(data=data)
	data = pp.remove_duplicates(data=data)
	
	very_high_card_cols, data = pp.remove_high_cardinality_cols(data=data, cardinality_thresh=0.999)
	(processed_data, label_encoder_mappings, reverse_label_encoder_mappings, freq_encoder_mappings,
	 reverse_freq_encoder_mappings, low_card_cols, high_card_cols, scaler) = pp.normalize_encode(data=data,
	                                                                                             cardinality_thresh=0.1,
	                                                                                             target_col=target_col)
	pp.export_data(data=processed_data, path="shared/", name='trm_processed_data')
	dump({"lem": label_encoder_mappings,
	      "rlem": reverse_label_encoder_mappings,
	      "fem": freq_encoder_mappings,
	      "rvfem": reverse_freq_encoder_mappings,
	      "lc": low_card_cols,
	      "hc": high_card_cols,
	      "s": scaler},
	     "shared/preprocessing_input.joblib")
	return processed_data


# FEATURE SELECTION  ---------------------------------------------------------------------------------------------------

def feature_selection_main(processed_data: pd.DataFrame, target: str) -> list:
	df_normal = processed_data[processed_data[target] == 0].copy()
	df_fraud = processed_data[processed_data[target] == 1].copy()
	logger_mlp.info(f' >>>>> Normal Samples <<<<<<')
	anova_normal_selected_features = fs.anova_features(data=df_normal, target_col=target, threshold=0.65, min_pct=0.5)
	mi_normal_features = fs.mutual_information(data=df_fraud, target_col=target, threshold=0.01, min_pct=0.6)
	
	logger_mlp.info(f' >>>>> Fraud Samples <<<<<<')
	anova_fraud_selected_features = fs.anova_features(data=df_normal, target_col=target, threshold=0.65, min_pct=0.5)
	mi_fraud_features = fs.mutual_information(data=df_fraud, target_col=target, threshold=0.01, min_pct=0.6)
	
	logger_mlp.info(f' >>>>> Embedded Samples <<<<<<')
	lgbm_features = fs.xgb_features(data=processed_data, target_col=target, threshold=0.3, min_pct=0.65)
	
	logger_mlp.info(f' >>>>> Features <<<<<<  ')
	normal_features = set(anova_normal_selected_features) | set(mi_normal_features)
	fraud_features = set(anova_fraud_selected_features) | set(mi_fraud_features)
	embedded_features = set(lgbm_features)
	
	main_features = None
	common_features = normal_features & fraud_features & embedded_features
	union_features = normal_features | fraud_features | embedded_features
	if common_features is not None:
		main_features = common_features
		logger_mlp.info(f"Common features({len(main_features)}):\n{main_features}")
	else:
		main_features = union_features
		logger_mlp.info(f"Common features({len(main_features)}):\n{main_features}")
	
	return list(main_features)


# ML MODEL  ------------------------------------------------------------------------------------------------------------
def ml_model(processed_data: pd.DataFrame, selected_features: list, target: str):
	supervised_models = {
		"DecisionTreeClassifier": DecisionTreeClassifier(criterion='entropy', max_depth=10, class_weight='balanced'),
		"RandomForestClassifier": RandomForestClassifier(n_estimators=50, max_depth=10, criterion='entropy',
		                                                 class_weight='balanced'),
		"XGBoostClassifier": XGBClassifier(n_estimators=50, max_depth=10),
		"LightGBMClassifier": LGBMClassifier(n_estimators=50, max_depth=10, verbosity=0)}
	
	X_train, X_val, X_test, y_train, y_val, y_test = mlmodels.split_data(data=processed_data, target_col=target,
	                                                                     selected_features=selected_features,
	                                                                     test_size=0.1, val_size=0.2)
	best_model, best_name, model_details = mlmodels.optimal_ML_Model(supervised_models=supervised_models,
	                                                                 X_train=X_train, y_train=y_train,
	                                                                 )
	logger_mlp.info(f' >>>>> Hyper parameter Tuning {best_name} <<<<<< ')
	best_model, best_parmas = mlmodels.hyperparameter_tuning(empty_model=best_model, X_grid=X_val, y_grid=y_val)
	mlmodels.evaluate_final_model(model=best_model, X_unseen=X_test, y_true=y_test)
	mlmodels.save_model(name=best_name, model=best_model, X=X_test)
	return best_model, best_parmas


# TIME -------------------------------------------------------------------------------------------------------------
def format_elapsed(start_time):
	elapsed = time.time() - start_time
	hours, rem = divmod(elapsed, 3600)
	minutes, seconds = divmod(rem, 60)
	logger_mlp.info(f"Elapsed time: {hours} hours, {minutes} minutes, {seconds} seconds")
	return hours, minutes, seconds


# MAIN MODEL TRAINING PHASE    -----------------------------------------------------------------------------------------
def main_model_extractor(data: pd.DataFrame, target: str):
	logger_mlp.info(f' {'=' * 50} PREPROCESSING START {'=' * 50} ')
	start_time = time.time()
	processed_data = pre_processing_training(data=data, target_col=target)
	preprocessing_hours, preprocessing_minutes, preprocessing_seconds = format_elapsed(start_time)
	logger_mlp.info(f' {'=' * 50}  END PREPROCESSING {'=' * 50} ')
	
	logger_mlp.info(f' {'=' * 50} FEATURE SELECTION START {'=' * 50} ')
	start_time = time.time()
	common_features = feature_selection_main(processed_data=processed_data, target=target)
	fselection_hours, fselection_minutes, fselection_seconds = format_elapsed(start_time)
	logger_mlp.info(f' {'=' * 50} END FEATURE SELECTION  {'=' * 50} ')
	
	logger_mlp.info(f' {'=' * 50} ML MODEL TRAIN START {'=' * 50} ')
	start_time = time.time()
	best_model, best_parmas = ml_model(processed_data=processed_data, selected_features=common_features, target=target)
	ml_model_hours, ml_model_minutes, ml_model_seconds = format_elapsed(start_time)
	logger_mlp.info(f' {'=' * 50} END ML MODEL TRAIN  {'=' * 50} ')
	
	time_dict = {
		"preprocessing_time": {"hours": preprocessing_hours, "minutes": preprocessing_minutes,
		                       "seconds": preprocessing_seconds},
		"feature_selection_time": {"hours": fselection_hours, "minutes": fselection_minutes,
		                           "seconds": fselection_seconds},
		"ml_model_time": {"hours": ml_model_hours, "minutes": ml_model_minutes, "seconds": ml_model_seconds}
	}
	
	return best_model, common_features, time_dict
