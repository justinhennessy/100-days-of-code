import pandas as pd
from scipy.stats import trim_mean, kurtosis
from fastai.imports import *
from fastai.structured import *
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
        array.append([column_name, mean, median, std, cv, trimmed_mean])
    print(tabulate(array,headers=['Column', 'Mean', 'Median', 'Std', 'cv', 'Trimmed Mean']))

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

def prepare_data(df_raw):
    # Sort data by date
    df_raw = df_raw.sort_values(by='licence_registration_date')

    # Convert annual_revenue from a string to a float
    df_raw['annual_revenue'] = pd.to_numeric(df_raw['annual_revenue'].str.replace(',', ''))

    # Convert fields to INT and setting any NaNs to the mean of that type
    case_types = ['cases_total','cases_open','cases_closed','cases_age_hours_total','cases_age_hours_average', 'last_login_days']

    for case_type in case_types:
        default_value = df_raw[case_type].fillna(df_raw[case_type].median())
        df_raw[case_type] = df_raw[case_type].fillna(default_value).astype(int)

    # Fix missing values for annual revenue, replace with mean/trimmed mean of the plan size they are on
    plan_list = df_raw.plan[~pd.isnull(df_raw.plan)].unique()

    for plan in plan_list:
        mean = round(df_raw.annual_revenue[df_raw.plan == plan].mean(), 2)
        trimmed_mean = trim_mean(df_raw.annual_revenue[df_raw.plan == plan].values, 0.1)

        if pd.isnull(mean):
            revenue = 0
        else:
            revenue = mean
        df_raw.loc[df_raw.plan==plan, 'annual_revenue'] = df_raw.loc[df_raw.plan==plan, 'annual_revenue'].fillna(revenue)

    # 'bin' last login days

    bins = [1, 3, 7, 14, 30, 60]
    group_names = ['day', 'few_days', 'week', 'fortnight', 'month']

    # need to get the mean of the plan size for last_login_days and set each row to that
    #df_raw.last_login_days = df_raw.last_login_days.fillna(np.mean(df_raw.last_login_days))

    last_login_categories = pd.cut(df_raw['last_login_days'], bins, labels=group_names)
    df_raw['last_login_categories'] = pd.cut(df_raw['last_login_days'], bins, labels=group_names)
    #pd.value_counts(df_raw['last_login_categories'])

    # one-hot encode fields
    dummy_columns = ['customer_account_status', 'last_login_categories', 'plan']

    for dummy_column in dummy_columns:
        dummy = pd.get_dummies(df_raw[dummy_column], prefix=dummy_column)
        df_raw = pd.concat([df_raw,dummy], axis=1)
        df_raw = df_raw.drop(columns=dummy_column)


    # This breaks all the date features up into number columns
    # These steps can only be run once then you need to comment them out
    add_datepart(df_raw, 'licence_registration_date')
    add_datepart(df_raw, 'golive_date')

    # Drop columns, some of these create "Data Leakage", some are just to test if it has impact when they are taken out
    df_raw = df_raw.drop(columns=['customer_account_status_Good', 'last_login_concern',
                                  'last_login_days', 'account_status', 'changing_platform',
                                  'new_platform', 'licence_status', 'canceldate',
                                  'cancel_details', 'cancel_reason'])

    # Set default values for NaN values in NPS
    df_raw.nps = df_raw.nps.fillna(np.nanmean(df_raw.nps))

    # Set NaN to zero
    features = ['churned', 'interactions_total', 'interactions_completed', 'interactions_no_response', 'interactions_no_onboarding', 'interactions_completed_training']

    for feature in features:
        df_raw[feature] = df_raw[feature].fillna(0)

    # Complete the transformation of all data into numbers using proc_df and create training dataframes
    train_cats(df_raw)

    return df_raw
