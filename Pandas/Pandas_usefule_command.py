import pandas as pd
#Creating a DataFrame


#Binning in Pandas


#Removing columns having a single constant value
const_cols = [c for c in train_df.columns if train_df[c].nunique(dropna=False)==1 ]



# Convert the value_counts to dict
df["Firmographics_Area"].value_counts().to_dict() #The output dict can be used in some other map function


#Using the transform function
train['TransactionAmt_to_mean_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('mean')



#Filling NA values
df["col_name_numerical"] = df["col_name_numerical"].fillna(0)
df["col_name_categorical"] = df["col_name_categorical"].fillna("Other")

#Carry_OHE (Pandas)
def carry_one_hot_encoding(df,columns):
    for col in columns:
      one_hot = pd.get_dummies(df[col],prefix=col)
      df.drop([col],axis=1,inplace=True)
      df = pd.concat([df,one_hot],axis=1)
    return df

def reducing_cardinality_of_categorical(df_train,df_val_filter_wo_used_prod,df_val,df_test,req_col=[],threshold=[]):
    for index,column in enumerate(req_col):
        distribution = df_train[column].value_counts(normalize=True)
        bottom_decile = distribution.quantile(q=threshold[index])
        less_freq_columns = distribution[distribution<bottom_decile].index.tolist()
        df_train.loc[df_train[column].isin(less_freq_columns),column] = "Others"


#Provides the feature importance of the trained model
def feature_important(model,col,segment,save_path):
  feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_,col)), columns=['Value','Feature'])
  feature_imp = feature_imp[["Feature","Value"]].sort_values(["Value"],ascending=[False])
  display(feature_imp)
  plt.figure(figsize=(20, 10))
  sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
  plt.title('Feature importance')
  plt.tight_layout()
  plt.show()
  plt.savefig(os.path.join(save_path,'feature_importances_'+segment+'.png'))



# Top-k value in pandas
k = 5
df = df.sort_values(["TPID","Pprop"],ascending=[True,False])
df["Rank"] = df.groupby(["TPID"]).cumcount()+1
df = df.loc[df["Rank"]<=k,:]




## Function to reduce the DF size
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


#Display the Name of columns, datatypes,Null values,Unique value
def resumetable(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Percentage'] = df.isnull().sum().values/ df.isnull().count().values    
    summary['Uniques'] = df.nunique().values
    return summary




def CalcOutliers(df_num): 

    # calculating mean and std of the array
    data_mean, data_std = np.mean(df_num), np.std(df_num)

    # seting the cut line to both higher and lower values
    # You can change this value
    cut = data_std * 3

    #Calculating the higher and lower cut values
    lower, upper = data_mean - cut, data_mean + cut

    # creating an array of lower, higher and total outlier values 
    outliers_lower = [x for x in df_num if x < lower]
    outliers_higher = [x for x in df_num if x > upper]
    outliers_total = [x for x in df_num if x < lower or x > upper]

    # array without outlier values
    outliers_removed = [x for x in df_num if x > lower and x < upper]
    
    print('Identified lowest outliers: %d' % len(outliers_lower)) # printing total number of values in lower cut of outliers
    print('Identified upper outliers: %d' % len(outliers_higher)) # printing total number of values in higher cut of outliers
    print('Total outlier observations: %d' % len(outliers_total)) # printing total number of values outliers of both sides
    print('Non-outlier observations: %d' % len(outliers_removed)) # printing total number of non outlier values
    print("Total percentual of Outliers: ", round((len(outliers_total) / len(outliers_removed) )*100, 4)) # Percentual of outliers in points
    
    return



#Finding the quantile
df = pd.DataFrame({"A":[1, 5, 3, 4, 2],
                   "B":[3, 2, 4, 3, 4],
                   "C":[2, 2, 7, 3, 4],
                   "D":[4, 3, 6, 12, 7]})
 
# using quantile() function to
# find the quantiles over the index axis
df.quantile([.1, .25, .5, .75], axis = 0)


