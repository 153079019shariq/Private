import pandas as pd
import numpy as np

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection


if __name__=="__main__":
  
  df = pd.read_csv("./data/train.csv")
  X = df.drop(["price_range"],axis=1).values
  y = df["price_range"].values


  classifier = ensemble.RandomForestClassifier(n_jobs=-1) 
  
  ###########################Grid_Search########################################
  """
  parameter_grid = {

                   "n_estimators" : [100,200,300,400],
                   "max_depth"    : [1,3,5,7],
                   "criterion"    : ["gini","entropy"],

                        }

  model = model_selection.GridSearchCV (
                                        estimator  = classifier, 
                                        param_grid = parameter_grid,
                                        scoring    = "accuracy", 
                                        n_jobs     = 1,
                                        verbose    = 10,
                                        cv         = 2
                                         )
  
  ############################################################################


  ###########################Random_Search###################################

  # Veru similar to Grid_Seach in terms of parameter except we need to specipy :
  # 1. Specify n_iter to say how many random values it will run
  # 2. The np.arange need to be passed instead of specific value 
  parameter_grid = {

                   "n_estimators" : np.arange(100,1500,100),
                   "max_depth"    : np.arange(1,20),
                   "criterion"    : ["gini","entropy"],

                        }

  model = model_selection.RandomizedSearchCV(
                                        estimator  = classifier, 
                                        param_distributions = parameter_grid,
                                        n_iter     = 10,                      
                                        scoring    = "accuracy", 
                                        n_jobs     = 1,
                                        verbose    = 10,
                                        cv         = 2
                                         )


  ########################################################################
  
  model.fit(X,y)
  
  print(model.best_score_)
  print(model.best_estimator_.get_params())
  """
