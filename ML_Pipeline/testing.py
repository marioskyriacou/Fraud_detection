import os
from Libraries import load, pd, logging

path = os.path.join(os.path.dirname(__file__), 'shared', 'preprocessing_input')
pre_preprocessing_dict = load(f'{path}.joblib')
lem, fem = pre_preprocessing_dict['lem'], pre_preprocessing_dict['fem']
scaler_ = pre_preprocessing_dict['s']
