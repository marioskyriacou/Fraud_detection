import numpy as np

import Train_Phase_Main as train_phase
import New_Samples_PreProcessingMain as n_sample_pp
from Libraries import pd, json
import Pre_Processing as pp
import TRM_XAI as xai
import os

# RUN THE TRAIN PIPELINE   -----------------------------------------------------------------------------------------
json_data = '''
[
  {
    "bankTransactionID": "4fa3208f-9e23-42dc-b330-844829d0c12c",
    "Transaction_Date": "23/01/2025",
    "Amount_converted": 32415.45,
    "Currency_Name": "INR",
    "From_Bank_Name": "Thiruvananthapuram Branch_Bank",
    "FromAccountNo": "d5f6ec07-d69e-4f47-b9b4-7c58ff17c19e_Savings",
    "From_Bank_Country": "Kerala",
    "From_Branch_Name": "Thiruvananthapuram Branch",
    "From_Branch_Country": "Kerala",
    "To_Bank_Name": "Restaurant_Network",
    "ToAccountNo": "214e03c5-5c34-40d1-a66c-f440aa2bbd02",
    "To_Bank_Country": "Thiruvananthapuram, Kerala",
    "To_Branch_Name": "Thiruvananthapuram, Kerala_Branch",
    "To_Branch_Country": "Thiruvananthapuram, Kerala",
    "hour": 16,
    "day_of_week": 3,
    "month": 1,
    "day_of_month":20,
    "is_weekend": true
  },
  {
    "bankTransactionID": "9b7d5f32-1a6e-4c72-bbb1-1234567890ab",
    "Transaction_Date": "23/01/2025",
    "Amount_converted": 45000.75,
    "Currency_Name": "INR",
    "From_Bank_Name": "Thiruvananthapuram Branch_Bank",
    "FromAccountNo": "d5f6ec07-d69e-4f47-b9b4-7c58ff17c19e_Savings",
    "From_Bank_Country": "Kerala",
    "From_Branch_Name": "Thiruvananthapuram Branch",
    "From_Branch_Country": "Kerala",
    "To_Bank_Name": "Restaurant_Network",
    "ToAccountNo": "214e03c5-5c34-40d1-a66c-f440aa2bbd02",
    "To_Bank_Country": "Thiruvananthapuram, Kerala",
    "To_Branch_Name": "Thiruvananthapuram, Kerala_Branch",
    "To_Branch_Country": "Thiruvananthapuram, Kerala",
    "hour": 18,
    "day_of_week": 3,
    "month": 1,
    "day_of_month": 24,
    "is_weekend": false
  }
]
'''

path = r'C:\eBOS\Work\Others\CharalambosKlitis_AI\TRM\ML_Pipeline\shared\trm_data_cleaned.csv'

# MAIN ML PIPELINE-----------------------------------------------------------------------------
data = pp.read_data_object(json_data)
data = pd.read_csv(path)
best_model, common_features, time_dict = train_phase.main_model_extractor(data=data, target='Is_Fraud')

# PREPROCESSED NEW SAMPLES -------------------------------------------------------------------
row_data = json.loads(json_data)
data = pd.DataFrame(row_data)
data_dict = data.to_dict('records')
processed_data_dict, pre_preprocessing_dict, model_dict = n_sample_pp.preprocessing_new_samples(data=data,
                                                                                                processing_joblib_path=os.path.join(
	                                                                                                os.path.dirname(__file__),
	                                                                                                'shared',
	                                                                                                'preprocessing_input'),
                                                                                                model_joblib_path=os.path.join(
	                                                                                                os.path.dirname(__file__),
	                                                                                                'shared',
	                                                                                                'best_model_details'))
xai_report = xai.explainability(model_dict, pre_preprocessing_dict, processed_data_dict, data_dict)
# print(xai_report)
