from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
def rfmodel(X,y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
  rf_Model = RandomForestClassifier()
  rf_Model.fit(X_train,y_train)
  print('Training Data AUC: = '+metrics.roc_auc_score(y_train.to_numpy(),rf_model.predict_proba(X_train),multi_class='ovr'))
  print('Test Data AUC: = '+metrics.roc_auc_score(y_test.to_numpy(),rf_model.predict_proba(X_test),multi_class='ovr')
  return rf_Model
