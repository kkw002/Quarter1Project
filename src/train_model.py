from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np

def model_gen(X,y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
  #Create baseline Model
  lr_model = LogisticRegression()

  rf_model = RandomForestClassifier()

  xgb_model = XGBClassifier() 

  rf_model.fit(X_train,y_train)
    
  xgb_model.fit(X_train,y_train)

  lr_model.fit(X_train,y_train)
  
  #Display AUC Baseline
    
  print('Baseline Train AUC: '+str(metrics.roc_auc_score(y_train.to_numpy(),lr_model.predict_proba(X_train)[:,1])))

  print('Baseline Test AUC: '+str(metrics.roc_auc_score(y_test.to_numpy(),lr_model.predict_proba(X_test)[:,1]))+'\n')
    
  fpr,tpr,threshholds = metrics.roc_curve(np.array(y_test),lr_model.predict_proba(X_test)[:,1])
  
  plt.plot(fpr,tpr,label='AUC= '+str(metrics.roc_auc_score(y_test.to_numpy(),lr_model.predict_proba(X_test)[:,1])))
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.legend()
  plt.savefig('auc_plots/baseline_auc_plot')
  plt.clf()
  #Display AUC for Random Forest

  print('Training Data Random Forest AUC:  '+str(metrics.roc_auc_score(y_train.to_numpy(),rf_model.predict_proba(X_train)[:,1])))

  print('Test Data Random Forest AUC: '+str(metrics.roc_auc_score(y_test.to_numpy(),rf_model.predict_proba(X_test)[:,1]))+'\n')
  
  print('Generating plot...\n')
  fpr,tpr,threshholds = metrics.roc_curve(np.array(y_test),rf_model.predict_proba(X_test)[:,1])
  plt.plot(fpr,tpr,label='AUC= '+str(metrics.roc_auc_score(y_test.to_numpy(),rf_model.predict_proba(X_test)[:,1])))
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.legend()
  plt.savefig('auc_plots/rf_auc_plot')
  plt.clf()

  #Displaying the AUC For XGB

  print('Training Data XGB AUC:  '+str(metrics.roc_auc_score(y_train.to_numpy(),xgb_model.predict_proba(X_train)[:,1])))
  print('Test Data XGB AUC: '+str(metrics.roc_auc_score(y_test.to_numpy(),xgb_model.predict_proba(X_test)[:,1]))+'\n')
  print('Generating plot...\n')
  fpr,tpr,threshholds = metrics.roc_curve(np.array(y_test),xgb_model.predict_proba(X_test)[:,1])
  plt.plot(fpr,tpr,label='AUC= '+str(metrics.roc_auc_score(y_test.to_numpy(),xgb_model.predict_proba(X_test)[:,1])))
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.legend()
  plt.savefig('auc_plots/xgb_auc_plot')
  plt.clf()

  #optimized model for Random Forest
  #param_grid = {'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 80, num = 10)],
  #             'max_features': ['auto', 'sqrt'],
  #             'max_depth': [2,4],
  #             'min_samples_split': [2, 5],
  #             'min_samples_leaf': [1, 2],
  #             'bootstrap': [True, False]}
  #rf_Model = RandomForestClassifier()
  #rf_random = RandomizedSearchCV(rf_Model,param_distributions=param_grid,n_iter=35,scoring='roc_auc',n_jobs=-1,cv=10,verbose=3)
  #rf_random.fit(X_train,y_train)
  hyperparam_model=RandomForestClassifier(n_estimators=64,min_samples_split=5,min_samples_leaf=1,max_features='auto',max_depth=4,bootstrap=True)
  hyperparam_model.fit(X_train,y_train)
  print('Training Data Optimized RandomForest AUC:  '+str(metrics.roc_auc_score(y_train.to_numpy(),hyperparam_model.predict_proba(X_train)[:,1])))
  print('Test Data Optimized RandomForest AUC:  '+str(metrics.roc_auc_score(y_test.to_numpy(),hyperparam_model.predict_proba(X_test)[:,1]))+'\n')
  print('Generating plot...\n')
  fpr,tpr,threshholds = metrics.roc_curve(np.array(y_test),hyperparam_model.predict_proba(X_test)[:,1])
  plt.plot(fpr,tpr,label='AUC= '+str(metrics.roc_auc_score(y_test.to_numpy(),hyperparam_model.predict_proba(X_test)[:,1])))
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.legend()
  plt.savefig('auc_plots/optimizedrf_auc_plot')
  plt.clf()
  #optimized model for XGBoost
  #params = {
  #  'learning_rate':[0.05,0.1,0.15,0.20,0.25],
  #  'max_depth':[2,4,5,6,8,10],
  #  'min_child_weight':[1,3,5,7],
  #  'gamma':[0.0,0.1,0.2,0.3,0.4],
  #  'colsample_bytree':[0.3,0.4,0.5,0.7]
  #}
  #classifier=XGBClassifier()
  #rsx= RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
  #rsx.fit(X_train,y_train)
  
  hxgb_model = XGBClassifier(min_child_weight=5,max_depth=2,learning_rate=0.2,gamma=0.4,colsample_bytree=0.3)
  hxgb_model.fit(X_train,y_train)

  print('Training Data Optimized XGB AUC: '+str(metrics.roc_auc_score(y_train.to_numpy(),hxgb_model.predict_proba(X_train)[:,1])))
  print('Test Data Optimized XGB AUC: '+str(metrics.roc_auc_score(y_test.to_numpy(),hxgb_model.predict_proba(X_test)[:,1]))+'\n')
  print('Generating plot...\n')
  fpr,tpr,threshholds = metrics.roc_curve(np.array(y_test),hxgb_model.predict_proba(X_test)[:,1])
  plt.plot(fpr,tpr,label='AUC= '+str(metrics.roc_auc_score(y_test.to_numpy(),hxgb_model.predict_proba(X_test)[:,1])))
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.legend()
  plt.savefig('auc_plots/optimizedxgb_auc_plot')
  plt.clf()
  return hyperparam_model,hxgb_model
