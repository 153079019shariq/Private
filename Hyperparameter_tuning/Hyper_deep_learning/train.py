import torch
import pandas as pd
import optuna
import utils
import numpy as np
DEVICE = "cpu"
EPOCHS = 30


def run_training(fold,params,save_model=False):

  df = pd.read_csv("data/train_features.csv") #Download the data from https://www.kaggle.com/competitions/lish-moa/data
  df = df.drop(["cp_type","cp_time","cp_dose"],axis=1)

  target_df = pd.read_csv("data/train_targets_fold.csv")
  features_columns  = df.drop("sig_id",axis=1).columns
  target_columns    = target_df.drop(["sig_id","kfold"],axis=1).columns

  df = df.merge(target_df,on="sig_id",how="left")
  print(df)

  train_df = df[df.kfold!=fold].reset_index(drop=True)
  valid_df = df[df.kfold!=fold].reset_index(drop=True)

  xtrain   = train_df[features_columns].to_numpy()
  ytrain   = train_df[target_columns].to_numpy()

  xvalid   = valid_df[features_columns].to_numpy()
  yvalid   = valid_df[target_columns].to_numpy()

  train_dataset = utils.MoaDataset(features=xtrain,target=ytrain)
  valid_dataset = utils.MoaDataset(features=xvalid,target=yvalid)

  train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=256,num_workers=4,shuffle=True)
  val_loader   = torch.utils.data.DataLoader(valid_dataset,batch_size=256,num_workers=4)

  model = utils.Model(nfeatures=xtrain.shape[1],
                     ntargets=ytrain.shape[1],
                     nlayers=params["num_layer"],
                     hidden_size=params["hidden_size"],
                     dropout=params["dropout"],
                     )

  model.to(DEVICE)
  optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
  eng = utils.Engine(model,optimizer,device=DEVICE)
  best_loss = 10000000
  early_stopping_iter = 10
  early_stopping_counter = 0

  for epoch in range(EPOCHS):
    train_loss = eng.train(train_loader)
    val_loss   = eng.train(val_loader)
    print(f"{fold},{train_loss},{val_loss}")
    if(best_loss > val_loss):
      best_loss =val_loss
      if(save_model):
        torch.save(model.state_dict(),f"model_{fold}.bin")

    else:
       early_stopping_counter += 1
    if(early_stopping_counter>early_stopping_iter):
      break
  return best_loss

    
def objective(trial):
 params = {
          "num_layer" : trial.suggest_int("num_layer",1,7),
          "hidden_size": trial.suggest_int("hidden_size",16,512),
          "dropout" : trial.suggest_uniform("dropout",0.1,0.7),
          "learning_rate" : trial.suggest_loguniform("learning_rate",1e-6,1e-3)

  }
 all_losses = []
 nfolds =1 #Initially nfolds was 5 but reduced it for faster computation
 for f_ in range(nfolds):  
   temp_loss = run_training(f_,params,save_model=False)
   all_losses.append(temp_loss)

 return np.mean(all_losses)




if __name__ == "__main__":
  #run_training(fold=0)
  study = optuna.create_study(direction="minimize")
  study.optimize(objective,n_trials=5)

  print("Best_trial")
  trial_ = study.best_trial
  print(trial_.values)
  print(trial_.params)

