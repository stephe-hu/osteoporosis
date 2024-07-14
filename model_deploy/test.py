import pandas as pd 
import pickle
from flask import Flask, render_template, request
from lime import lime_tabular
import os
from sklearn.preprocessing import StandardScaler
import numpy as np
import unittest

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = 'C:/Users/16462/python-projects/osteoporosis/model_dev/dataset/processed'
# model_dir = os.path.join(current_dir, 'model_dev', 'models')
model_dir = 'C:/Users/16462/python-projects/osteoporosis/model_dev/models'

# load in values for ordinal encoding
mapping_age = pd.read_csv(os.path.join(data_dir, 'mapping_age.csv'))
mapping_alcohol_consumption = pd.read_csv(os.path.join(data_dir, 'mapping_alcohol_consumption.csv'))
mapping_body_weight = pd.read_csv(os.path.join(data_dir, 'mapping_body_weight.csv'))
mapping_calcium_intake = pd.read_csv(os.path.join(data_dir, 'mapping_calcium_intake.csv'))
maping_family_history = pd.read_csv(os.path.join(data_dir, 'mapping_family_history.csv'))
mapping_gender = pd.read_csv(os.path.join(data_dir, 'mapping_gender.csv'))
mapping_hormonal_changes = pd.read_csv(os.path.join(data_dir, 'mapping_hormonal_changes.csv'))
mapping_medical_conditions = pd.read_csv(os.path.join(data_dir, 'mapping_medical_conditions.csv'))
mapping_medications = pd.read_csv(os.path.join(data_dir, 'mapping_medications.csv'))
mapping_osteoporosis = pd.read_csv(os.path.join(data_dir, 'mapping_osteoporosis.csv'))
mapping_physical_activity = pd.read_csv(os.path.join(data_dir, 'mapping_physical_activity.csv'))
mapping_prior_fractures = pd.read_csv(os.path.join(data_dir, 'mapping_prior_fractures.csv'))
mapping_race_ethnicity = pd.read_csv(os.path.join(data_dir, 'mapping_race_ethnicity.csv'))
mapping_smoking = pd.read_csv(os.path.join(data_dir, 'mapping_smoking.csv'))
mapping_vitamin_d_intake = pd.read_csv(os.path.join(data_dir, 'mapping_vitamin_d_intake.csv'))


mapping_age_list = mapping_age['age'].tolist()
mapping_alcohol_consumption_list = mapping_alcohol_consumption['alcohol_consumption'].tolist()
mapping_body_weight_list = mapping_body_weight['body_weight'].tolist()
mapping_calcium_intake_list = mapping_calcium_intake['calcium_intake'].tolist()
mapping_family_history_list = maping_family_history['family_history'].tolist()
mapping_gender_list = mapping_gender['gender'].tolist()
mapping_hormonal_changes_list = mapping_hormonal_changes['hormonal_changes'].tolist()
mapping_medical_conditions_list = mapping_medical_conditions['medical_conditions'].tolist()
mapping_medications_list = mapping_medications['medications'].tolist()
mapping_osteoporosis_list = mapping_osteoporosis['osteoporosis'].tolist()
mapping_physical_activity_list = mapping_physical_activity['physical_activity'].tolist()
mapping_prior_fractures_list = mapping_prior_fractures['prior_fractures'].tolist()
mapping_race_ethnicity_list = mapping_race_ethnicity['race_ethnicity'].tolist()
mapping_smoking_list = mapping_smoking['smoking'].tolist()
mapping_vitamin_d_intake_list = mapping_vitamin_d_intake['vitamin_d_intake'].tolist()


# load in the model, scaler
scaler_path = os.path.join(model_dir, 'osteoporosis_scalar.sav')
loaded_scaler = pickle.load(open(scaler_path, 'rb'))
model_path = os.path.join(model_dir, 'xgboost_model.sav')
loaded_model = pickle.load(open(model_path, 'rb'))
x_train_path = os.path.join(model_dir, 'X_train.sav')
loaded_X_train = pickle.load(open(x_train_path, 'rb'))
x_columns_path = os.path.join(model_dir, 'X_columns.sav')
loaded_X_columns = pickle.load(open(x_columns_path, 'rb'))


class TestMappings(unittest.TestCase):
    def test_mappings(self):
    # Mock data
        hormonal_changes = 'Normal'  # or 'Postmenopausal'

    # Add the assertion here
        assert hormonal_changes in mapping_hormonal_changes['hormonal_changes'].values, f"{hormonal_changes} not found in 'hormonal_changes' column"

    # Then run the lines of code
        hormonal_changes = mapping_hormonal_changes[mapping_hormonal_changes['hormonal_changes'] == hormonal_changes]['hormonal_changes_ordinal'].values[0]
    # ...

    # Check if the output is as expected
        self.assertEqual(hormonal_changes, 0)  # replace 'expected_value' with 0
    # ...
 
if __name__ == '__main__':
    unittest.main()



