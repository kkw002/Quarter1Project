from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
def holdout_results(rmodel,xmodel,X,y):
  print('Holdout Random Forest AUC: '+metrics.roc_auc_score(y.to_numpy(),rmodel.predict_proba(X),multi_class='ovr',average='weighted'))
  print('Holdout XGB AUC: '+metrics.roc_auc_score(y.to_numpy(),xmodel.predict_proba(X),multi_class='ovr',average='weighted'))
