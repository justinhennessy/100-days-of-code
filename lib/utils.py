import pandas as pd
from scipy.stats import trim_mean, kurtosis
from fastai.imports import *
from fastai.structured import *
from tabulate import tabulate
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
from sklearn.metrics import f1_score,\
    accuracy_score, confusion_matrix,\
    precision_score, recall_score,\
    roc_curve, roc_auc_score,\
    cohen_kappa_score, mean_absolute_error,\
    precision_recall_curve, auc,\
    average_precision_score

# Constants
PROBABILITY_CUTOFF = 0.50

def split_vals(a,n): return a[:n], a[n:]

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
    #print("precision, recall, f1_score, accuracy, cohen_kappa_score, mean abs error")
    print(precision_score(y_valid,validate_predictions), recall_score(y_valid,validate_predictions), f1_score(y_valid,validate_predictions), accuracy_score(y_valid,validate_predictions), cohen_kappa_score(y_valid,validate_predictions), mean_absolute_error(y_valid,validate_predictions))

def graph_corr(data_frame):
    fig, ax = plt.subplots(figsize=(20,10))
    corr = data_frame.corr()
    sns.heatmap(corr, cmap='YlGnBu', annot_kws={'size':30}, ax=ax)
    ax.set_title("Correlation Matrix", fontsize=14)
    plt.show()

def data_summary_feature(data_frame, feature):
    """
    Doc string
    """

    max_value = data_frame[feature].max()
    min_value = data_frame[feature].min()
    mean = data_frame[feature].mean()
    median = data_frame[feature].median()
    std = data_frame[feature].std()
    co_variant = data_frame[feature].std()/data_frame[feature].mean()
    trimmed_mean = trim_mean(data_frame[feature].values, 0.1)
    return [feature, min_value, max_value, mean, trimmed_mean, median, std, co_variant]

def data_summary_dataframe(data_frame):
    array = []
    for feature in data_frame.select_dtypes(include=['float64', 'int64']).columns:
        array.append(data_summary_feature(data_frame, feature))

    print(tabulate(array, headers=['Column', 'Min', 'Max',
                                   'Mean', 'Trimmed Mean',
                                   'Median', 'Std', 'cv'], tablefmt="simple"))

def conf_matrix(y_valid, validate_predictions):
    confusion_matrix_data = confusion_matrix(y_valid, validate_predictions)
    print(confusion_matrix_data)
