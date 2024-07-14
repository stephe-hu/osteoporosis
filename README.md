# Osteoporosis

## Context
The dataset is obtained from [Kaggle](https://www.kaggle.com/datasets/amitvkulkarni/lifestyle-factors-influencing-osteoporosis). The purpose of this dataset is to predict the risk of osteopororsis. In this project, EDA is first performed to understand the dataset and the correlation between the features. Then, the dataset is used to train models to predict the risk of osteoporosis. The models are evaluated based on AUC, accuracy, recall, precision, and prevalence. The best model is saved via `pickel`. `SHAP` and `LIME` are used to explain the model predictions.
The model is then deployed using `Flask`. Results can be seen in [here](./model_deploy/images).