import pandas as pd
import numpy as np
from hyperopt import hp
import numpy as np
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from hyperopt.pyll import scope


import lightgbm as lgb
import xgboost as xgb
#import catboost as ctb
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials
from sklearn import metrics


# XGB parameters
xgb_reg_params = {
    #'learning_rate':    hp.choice('learning_rate',  [0.1,0.01,0.3]),
    #'subsample':        hp.uniform('subsample', 0.8, 1),
   
    'max_depth': scope.int(hp.quniform("max_depth", 3, 30, 1)),
    'gamma': hp.uniform ('gamma', 1,9),
    'reg_alpha' : scope.int(hp.quniform('reg_alpha', 4,180,1)),
    'reg_lambda' : hp.uniform('reg_lambda', 0,1),
    'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
    'min_child_weight' : scope.int(hp.quniform('min_child_weight', 0, 100, 1)),
    "scale_pos_weight" : hp.choice("scale_pos_weight",[1,10,50,100,1000]),
    'seed': 0,
    'n_estimators':     100,
}
xgb_fit_params = {
    'eval_metric': 'auc',
    'early_stopping_rounds': 10,
    #'verbose': -1
}
xgb_para = dict()
xgb_para['reg_params'] = xgb_reg_params
xgb_para['fit_params'] = xgb_fit_params





# LightGBM parameters
lgb_reg_params = {
    "scale_pos_weight" : hp.uniform('scale_pos_weight', 1,1000),
    'learning_rate':    hp.choice('learning_rate',    [0.1,0.01,0.3]),
    #'max_depth':        scope.int(hp.quniform("max_depth", 3, 30, 1)),
    "num_leaves" :      scope.int(hp.quniform("num_leaves", 20, 150, 1)), 
    'min_child_weight': scope.int(hp.quniform('min_child_weight', 0, 100, 1)),
    'colsample_bytree':  hp.uniform('colsample_bytree', 0.2,1),
    'reg_alpha' :     hp.uniform('reg_alpha', 0,20),
    'reg_lambda' :    hp.uniform('reg_lambda', 0,1),
    'subsample':        hp.uniform('subsample', 0.8, 1),
    'n_estimators':     100,
}
lgb_fit_params = {
    #'eval_metric': 'auc',
    #'early_stopping_rounds': 10,
    'verbose':-1 
}
lgb_para = dict()
lgb_para['reg_params'] = lgb_reg_params
lgb_para['fit_params'] = lgb_fit_params

"""
# CatBoost parameters
ctb_reg_params = {
    'learning_rate':     hp.choice('learning_rate',     np.arange(0.05, 0.31, 0.05)),
    'max_depth':         hp.choice('max_depth',         np.arange(5, 16, 1, dtype=int)),
    'colsample_bylevel': hp.choice('colsample_bylevel', np.arange(0.3, 0.8, 0.1)),
    'n_estimators':      100,
    'eval_metric':       'RMSE',
}
ctb_fit_params = {
    'early_stopping_rounds': 10,
    'verbose': False
}
ctb_para = dict()
ctb_para['reg_params'] = ctb_reg_params
ctb_para['fit_params'] = ctb_fit_params
"""

# COMMAND ----------



# COMMAND ----------


class HPOpt(object):

    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test  = x_test
        self.y_train = y_train
        self.y_test  = y_test
      
    def process(self, fn_name, space, trials, algo, max_evals):
        fn = getattr(self, fn_name)
        try:
            result = fmin(fn=fn, space=space, algo=algo, max_evals=max_evals, trials=trials)
            best_params = space_eval(space["reg_params"], result)
            print("Best_hyperparameter  ",best_params) 
        except Exception as e:
            return {'status': STATUS_FAIL, 'exception': str(e)}
        return result, trials

    def xgb_cls(self, para):
        reg = xgb.XGBClassifier(**para['reg_params'])
        return self.train_reg(reg, para)
      
    def lgb_cls(self, para):
        reg = lgb.LGBMClassifier(**para['reg_params'])
        return self.train_reg(reg, para)
  
    def ctb_reg(self, para):
        reg = ctb.CatBoostRegressor(**para['reg_params'])
        return self.train_reg(reg, para)
  
    def train_reg(self, reg, para):
        reg.set_params(**para['fit_params'])
        reg.fit(self.x_train, self.y_train,
                eval_set= [(self.x_test, self.y_test)],verbose=0)  
        y_pred_probab_train = reg.predict_proba(self.x_train)[:, 1]    
        y_pred_probab = reg.predict_proba(self.x_test)[:, 1]    
        
        train_auc = metrics.roc_auc_score(self.y_train, y_pred_probab_train,multi_class="ovr")
        val_auc = metrics.roc_auc_score(self.y_test, y_pred_probab,multi_class="ovr")
        
        print(f"Training_AUC {train_auc}  Validation_AUC {val_auc} Parameter {para['reg_params']}")
        return {'loss': -val_auc, 'status': STATUS_OK}
      
if __name__=="__main__":

  df = pd.read_csv("./data/train.csv")
  X = df.drop(["price_range"],axis=1).values
  y = (df["price_range"] >2).astype(int) 
  print(np.unique(y))


  X_train, X_test, y_train, y_test =train_test_split(X,y,random_state=104,test_size=0.2, shuffle=True)

  obj = HPOpt(X_train, X_test, y_train, y_test)
  xgb_opt = obj.process(fn_name='xgb_cls', space=xgb_para, trials=Trials(), algo=tpe.suggest, max_evals=8)
  #lgb_opt = obj.process(fn_name='lgb_cls', space=lgb_para, trials=Trials(), algo=tpe.suggest, max_evals=150)
  #ctb_opt = obj.process(fn_name='ctb_reg', space=ctb_para, trials=Trials(), algo=tpe.suggest, max_evals=100)


# COMMAND ----------



"""
# COMMAND ----------

reg_params = {'colsample_bytree': 0.405430275199931, 'learning_rate': 0.1, 'min_child_weight': 53, 'n_estimators': 100, 'num_leaves': 106, 'reg_alpha': 2.2450413294821288, 'reg_lambda': 0.8194283820257628, 'subsample': 0.9346754845675219}
fit_params = {
    'eval_metric': 'auc',
    'early_stopping_rounds': 10,
    'verbose': False
}
lgb_cls = lgb.LGBMClassifier(**reg_params)
lgb_cls.fit(X_train, y_train,
                eval_set= [(X_test, y_test)], **fit_params)

y_pred_probab_train = lgb_cls.predict_proba(X_train)[:, 1]    
y_pred_probab = lgb_cls.predict_proba(X_test)[:, 1]    
        
train_auc = metrics.roc_auc_score(y_train, y_pred_probab_train)
val_auc = metrics.roc_auc_score(y_test, y_pred_probab)

print(f"Training_AUC {train_auc}  Validation_AUC {val_auc} ")

# COMMAND ----------

xgb_opt = obj.process(fn_name='xgb_cls', space=xgb_para, trials=Trials(), algo=tpe.suggest, max_evals=8)

"""
"""
#http://hyperopt.github.io/hyperopt/getting-started/minimizing_functions/
Check the below items
 trials.trials - a list of dictionaries representing everything about the search
trials.results - a list of dictionaries returned by 'objective' during the search
trials.losses() - a list of losses (float for each 'ok' trial)
trials.statuses() - a list of status strings
"""
