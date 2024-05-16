import pandas as pd 
import pickle
from flask import Flask, render_template, request
from lime import lime_tabular
import os
from sklearn.preprocessing import StandardScaler
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = 'C:/Users/16462/python-projects/osteoporosis/model_dev/dataset/processed'
# model_dir = os.path.join(current_dir, 'model_dev', 'models')
model_dir = 'C:/Users/16462/python-projects/osteoporosis/model_dev/models'

## load in values for ordinal encoding
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


## load in the model, scaler
scaler_path = os.path.join(model_dir, 'osteoporosis_scalar.sav')
loaded_scaler = pickle.load(open(scaler_path, 'rb'))
model_path = os.path.join(model_dir, 'xgboost_model.sav')
loaded_model = pickle.load(open(model_path, 'rb'))
x_train_path = os.path.join(model_dir, 'X_train.sav')
loaded_X_train = pickle.load(open(x_train_path, 'rb'))
x_columns_path = os.path.join(model_dir, 'X_columns.sav')
loaded_X_columns = pickle.load(open(x_columns_path, 'rb'))

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    ## if request is post, then get the values from the form
    if request.method == 'POST':
        print('request.form:', request.form)

        age = request.form['age']
        alcohol_consumption = request.form['alcohol_consumption']
        body_weight = request.form['body_weight']
        calcium_intake = request.form['calcium_intake']
        family_history = request.form['family_history']
        gender = request.form['gender']
        hormonal_changes = request.form['hormonal_changes']
        medical_conditions = request.form['medical_conditions']
        medications = request.form['medications']
        physical_activity = request.form['physical_activity']
        prior_fractures = request.form['prior_fractures']
        race_ethnicity = request.form['race_ethnicity']
        smoking = request.form['smoking']
        vitamin_d_intake = request.form['vitamin_d_intake']

        print('age:', age)
        print('alcohol_consumption:', alcohol_consumption)
        print('body_weight:', body_weight)
        print('calcium_intake:', calcium_intake)
        print('family_history:', family_history)
        print('gender:', gender)
        print('hormonal_changes:', hormonal_changes)
        print('medical_conditions:', medical_conditions)
        print('medications:', medications)
        print('physical_activity:', physical_activity)
        print('prior_fractures:', prior_fractures)
        print('race ethnicity:', race_ethnicity)
        print('smoking:', smoking)
        print('vitamin_d_intake:', vitamin_d_intake)

        ## create a non-scaled df
        df_nonscaled = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'hormonal_changes': [hormonal_changes],
            'family_history': [family_history],
            'race/ethnicity': [race_ethnicity],
            'body_weight': [body_weight],
            'calcium_intake': [calcium_intake],
            'vitamin_d_intake': [vitamin_d_intake],
            'physical_activity': [physical_activity],
            'smoking': [smoking],
            'alcohol_consumption': [alcohol_consumption],
            'medical_conditions': [medical_conditions],
            'medications': [medications],
            'prior_fractures': [prior_fractures],
        })

        ## based on the values, get the ordinal encoding
        age = mapping_age[mapping_age['age'] == age]['age_ordinal'].values[0] # noqa
        print('age:', age)
        
        alcohol_consumption = mapping_alcohol_consumption[mapping_alcohol_consumption['alcohol_consumption'] == alcohol_consumption]['alcohol_consumption_ordinal'].values[0] # noqa 
        print('alcohol_consumption:', alcohol_consumption)

        body_weight = mapping_body_weight[mapping_body_weight['body_weight'] == body_weight]['body_weight_ordinal'].values[0] # noqa
        print('body_weight:', body_weight)

        calcium_intake = mapping_calcium_intake[mapping_calcium_intake['calcium_intake'] == calcium_intake]['calcium_intake_ordinal'].values[0] # noqa
        print('calcium_intake:', calcium_intake)

        family_history = maping_family_history[maping_family_history['family_history'] == family_history]['family_history_ordinal'].values[0] # noqa
        print('family_history:', family_history)

        gender = mapping_gender[mapping_gender['gender'] == gender]['gender_ordinal'].values[0] # noqa
        print('gender:', gender)

        hormonal_changes = mapping_hormonal_changes[mapping_hormonal_changes['hormonal_changes'] == hormonal_changes]['hormonal_changes_ordinal'].values[0] # noqa
        print('hormonal_changes:', hormonal_changes)
        
        medical_conditions = mapping_medical_conditions[mapping_medical_conditions['medical_conditions'] == medical_conditions]['medical_conditions_ordinal'].values[0] # noqa
        print('medical_conditions:', medical_conditions)
        
        medications = mapping_medications[mapping_medications['medications'] == medications]['medications_ordinal'].values[0] # noqa
        print('medications:', medications)
                
        physical_activity = mapping_physical_activity[mapping_physical_activity['physical_activity'] == physical_activity]['physical_activity_ordinal'].values[0] # noqa
        print('physical_activity:', physical_activity)
        
        prior_fractures = mapping_prior_fractures[mapping_prior_fractures['prior_fractures'] == prior_fractures]['prior_fractures_ordinal'].values[0] # noqa
        print('prior_fractures:', prior_fractures)
        
        race_ethnicity = mapping_race_ethnicity[mapping_race_ethnicity['race_ethnicity'] == race_ethnicity]['race_ethnicity_ordinal'].values[0] # noqa
        print('race_ethnicity:', race_ethnicity)
        
        smoking = mapping_smoking[mapping_smoking['smoking'] == smoking]['smoking_ordinal'].values[0] # noqa
        print('smoking:', smoking)
        
        vitamin_d_intake = mapping_vitamin_d_intake[mapping_vitamin_d_intake['vitamin_d_intake'] == vitamin_d_intake]['vitamin_d_intake_ordinal'].values[0] # noqa
        print('vitamin_d_intake:', vitamin_d_intake)
        
        
        

        ## create a dataframe with the values
        df = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'hormonal_changes': [hormonal_changes],
            'family_history': [family_history],
            'race/ethnicity': [race_ethnicity],
            'body_weight': [body_weight],
            'calcium_intake': [calcium_intake],
            'vitamin_d_intake': [vitamin_d_intake],
            'physical_activity': [physical_activity],
            'smoking': [smoking],
            'alcohol_consumption': [alcohol_consumption],
            'medical_conditions': [medical_conditions],
            'medications': [medications],
            'prior_fractures': [prior_fractures],
        })

        print('df:', df)

        ## scale the values
        df_scaled = loaded_scaler.transform(df)
        print('df_scaled:', df_scaled)
        
        ## make the prediction
        prediction = loaded_model.predict(df_scaled)
        print('ML PREDICTION: ', prediction[0])

        ## map the prediction to a string
        if prediction[0] == 0:
            prediction = 'No osteoporosis'
        else:
            prediction = 'Osteoporosis'
        
        ## generate the explanation
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=loaded_X_train,
            feature_names=loaded_X_columns,
            class_names=['no osteoporosis', 'osteoporosis'],
            mode='classification',
        )

        ## drop the inner list
        df_scaled_ = df_scaled[0]
        print('df_scaled_:', df_scaled_)
        exp = explainer.explain_instance(df_scaled_, loaded_model.predict_proba, num_features=9)
        exp_html = exp.as_html()
  
        ## conver df_nonscaled to dict
        df_nonscaled = df_nonscaled.to_dict('records')
        df_nonscaled = df_nonscaled[0]
        print('df_nonscaled:', df_nonscaled)


        ## return the prediction
        return render_template(
            'index.html',
            prediction=prediction,
            df_nonscaled=df_nonscaled,
            exp_html=exp_html,
            age_list=mapping_age_list,
            alcohol_consumption_list=mapping_alcohol_consumption_list,
            body_weight_list=mapping_body_weight_list,
            calcium_intake_list=mapping_calcium_intake_list,
            family_history_list=mapping_family_history_list,
            gender_list=mapping_gender_list,
            hormonal_changes_list=mapping_hormonal_changes_list,
            medical_conditions_list=mapping_medical_conditions_list,
            medications_list=mapping_medications_list,
            osteoporosis_list=mapping_osteoporosis_list,
            physical_activity_list=mapping_physical_activity_list,
            prior_fractures_list=mapping_prior_fractures_list,
            race_ethnicity_list=mapping_race_ethnicity_list,
            smoking_list=mapping_smoking_list,
            vitamin_d_intake_list=mapping_vitamin_d_intake_list,
        )

    else:
        return render_template(
            'index.html',
            age_list=mapping_age_list,
            alcohol_consumption_list=mapping_alcohol_consumption_list,
            body_weight_list=mapping_body_weight_list,
            calcium_intake_list=mapping_calcium_intake_list,
            family_history_list=mapping_family_history_list,
            gender_list=mapping_gender_list,
            hormonal_changes_list=mapping_hormonal_changes_list,
            medical_conditions_list=mapping_medical_conditions_list,
            medications_list=mapping_medications_list,
            osteoporosis_list=mapping_osteoporosis_list,
            physical_activity_list=mapping_physical_activity_list,
            prior_fractures_list=mapping_prior_fractures_list,
            race_ethnicity_list=mapping_race_ethnicity_list,
            smoking_list=mapping_smoking_list,
            vitamin_d_intake_list=mapping_vitamin_d_intake_list,
        )
        

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        debug=True, 
        port=5001)