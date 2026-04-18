import pandas as pd
import json
import numpy as np
import os
import tqdm
import time
import logging

import seaborn as sns
import matplotlib.pyplot as plt

from typing import Tuple, List, Union, Dict, Any, Optional
from joblib import dump, load
from collections import Counter

from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_regression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import (
	classification_report,
	accuracy_score,
	precision_score,
	recall_score,
	f1_score,
	confusion_matrix,
	roc_auc_score,
	precision_recall_curve,
	balanced_accuracy_score,
	auc,
	roc_curve)

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)

logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
