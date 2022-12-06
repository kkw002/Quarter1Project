# DSC180A Quarter 1 Project: Predicting Deliquency of Credit Applicants
This project explores relevant attributes in predicting future delinquencies utlizing models on datasets given, one containing purely training data and another being the holdout set. The classification models used in this project are XGBoost,Random Forest Classifier and Logistic Regression.

## Accessing Training and Holdout data

The training data can be accessed through this google link: https://drive.google.com/file/d/1_C3Hx2s0YJ0rTdOPNVvufA1bRH5eRrPO/view?usp=sharing

The holdout set can be accessed through this google link: https://drive.google.com/file/d/1_C3Hx2s0YJ0rTdOPNVvufA1bRH5eRrPO/view?usp=sharing

(1) Download the data from the links, copy the filepath in which they are stored.

(2) In their respective config files, replace outdir with the filepath of each file.
## Running on Actual Data
When launching the docker image, include ```-m 32 -c 8```after the docker image.

## Viewing Results
To get the data from the config files, run ``` python run.py data ```

To view the roc_auc scores of the models, run  ``` python run.py results ```

To check if the models work, run ``` python run.py test ```
