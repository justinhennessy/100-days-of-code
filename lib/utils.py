import pandas as pd
from scipy.stats import trim_mean, kurtosis
from fastai.imports import *
from fastai.structured import *
from tabulate import tabulate
from sklearn.metrics import f1_score,\
    accuracy_score, confusion_matrix,\
    precision_score, recall_score,\
    roc_curve, roc_auc_score,\
    cohen_kappa_score, mean_absolute_error,\
    precision_recall_curve, auc,\
    average_precision_score

# Constants
PROBABILITY_CUTOFF = 0.50

# Utility functions
def plot_roc_pr(m, X_valid, y_valid):
    # Generate the probabilities
    #y_pred_prob = generate_predictions(X_valid)
    y_pred_prob = m.predict_proba(X_valid)
    y_pred_prob = y_pred_prob[:, 1]

    # Calculate the roc metrics
    fpr, tpr, thresholds = roc_curve(y_valid, y_pred_prob)

    fig, ax = plt.subplots(figsize=(20,10))

    # Plot the ROC curve
    plt.plot(fpr,tpr, label='ROC')

    # Add labels and diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], "k--", label='no skill')

    # Plot a precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_valid, y_pred_prob)
    area_under_curve = auc(recall, precision)
    ap = average_precision_score(y_valid, y_pred_prob)

    # plot no skill
    pyplot.plot([0, 1], [0.5, 0.5], linestyle='--', label='no skill')
    # plot the precision-recall curve for the model
    pyplot.plot(recall, precision, marker='.', label='precision-recall')

    legend = ax.legend(loc='best', shadow=True, fontsize='medium')

    # show the plot
    pyplot.show()
    
    # Output AUC and average precision score
    print('auc=%.3f ap=%.3f' % (area_under_curve, ap))
    
def uber_score(y_valid, validate_predictions):
    print("precision, recall, f1_score, accuracy, cohen_kappa_score, mean abs error")
    return [precision_score(y_valid,validate_predictions), recall_score(y_valid,validate_predictions), f1_score(y_valid,validate_predictions), accuracy_score(y_valid,validate_predictions), cohen_kappa_score(y_valid,validate_predictions), mean_absolute_error(y_valid,validate_predictions)]

def graph_corr(df):
    fig, ax = plt.subplots(figsize=(20,10))
    corr = df.corr()
    sns.heatmap(corr, cmap='YlGnBu', annot_kws={'size':30}, ax=ax)
    ax.set_title("Correlation Matrix", fontsize=14)
    plt.show()
    
def data_summary(df_raw):
    array = []
    for column_name in df_raw.select_dtypes(include=['float64', 'int64']).columns:
        mean = df_raw[column_name].mean()
        median = df_raw[column_name].median()
        std = df_raw[column_name].std()
        cv = df_raw[column_name].std()/df_raw[column_name].mean()
        trimmed_mean = trim_mean(df_raw[column_name].values, 0.1)
        array.append([column_name, mean, trimmed_mean, median, std, cv])
    print(tabulate(array,headers=['Column', 'Mean', 'Trimmed Mean', 'Median', 'Std', 'cv']))

def generate_predictions(X_valid, cutoff=PROBABILITY_CUTOFF):
    return m.predict(X_valid)
    #return (m.predict_proba(X_valid)[:,1] >= cutoff).astype(bool)
    
def conf_matrix(y_valid, validate_predictions):
    cm = confusion_matrix(y_valid, validate_predictions)
    print(cm)
    
def remove_columns_test(df):
    m = RandomForestClassifier(
        n_estimators=10,
        min_samples_leaf=1, 
        max_features='sqrt',
        n_jobs=-1, 
        #oob_score=True,
        max_depth=3,
        bootstrap=False,
        criterion='gini',
        class_weight={0: 2, 1: 1})
    x, _ = split_vals(df, n_trn)
    m.fit(x, y_train)
    y_pred = m.predict(x)
    return uber_score(y_train, y_pred)

