import pandas as pd
def lift_curve(y_true, y_score):

    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    y_true = (y_true == 1)
    #1.Sort the y_true  w.r.t y_score in decreasing order of probability.
    sorted_indices = np.argsort(y_score)[::-1]
    y_true = y_true[sorted_indices]
    #2. Find the cumulative sum.
    gains = np.cumsum(y_true)
    Num_sample = np.arange(start=1, stop=len(y_true) + 1)
    #3.Normalize it so that min value is 0 and max value is 1.
    gains = gains / float(np.sum(y_true))
    #4.Find the percentage of sample.
    percentages = Num_sample / float(len(y_true))
    lift = gains / percentages
    df = pd.DataFrame({"Percent_of_sample":percentages,"Lift_curve":lift})
    return df
def plot_lift_curve(df, figsize=None):
    percentages, lift = df["Percent_of_sample"].values,df["Lift_curve"].values
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title('Lift Curve', fontsize="large")
    ax.plot(percentages, lift, lw=3, label='Model')
    ax.plot([0, 1], [1, 1], 'k--', lw=2, label='Baseline')
    ax.set_xlabel('Percentage of sample', fontsize=12)
    ax.set_ylabel('Lift', fontsize=12)
    ax.tick_params(labelsize=12)
    ax.grid('on')
    ax.legend(loc='lower right', fontsize=12)
    return ax

df = lift_curve(y_true=y_val2, y_score=lgb_cls.predict_proba(X_val2)[:, 1])
plot_lift_curve(df,figsize=[15,10])
display(df)
print(df["Lift_curve"].max())
