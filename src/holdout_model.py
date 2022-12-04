from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
def holdout_results(model,X,y):
  result_string = 'Holdout AUC: '+metrics.roc_auc_score(y.to_numpy(),hyperparam_model.predict_proba(X),multi_class='ovr',average='weighted'))
  return result_string
