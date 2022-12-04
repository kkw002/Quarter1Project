from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
def rfmodel(X,y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
  hyperparam_model=RandomForestClassifier(n_estimators=48,min_samples_split=2,min_samples_leaf=2,max_features='auto',max_depth=2,bootstrap=True)
  hyperparam_model.fit(X_train,y_train)
  print('Training Data AUC: = '+metrics.roc_auc_score(y_train.to_numpy(),hyperparam_model.predict_proba(X_train),multi_class='ovr',average='weighted'))
  print('Test Data AUC: = '+metrics.roc_auc_score(y_test.to_numpy(),hyperparam_model.predict_proba(X_test),multi_class='ovr',average='weighted'))
  return hyperparam_model
