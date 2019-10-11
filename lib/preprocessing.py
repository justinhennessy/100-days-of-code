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

def logify_feature(df, feature):
    df[feature + '_log'] = np.log(df[feature])
    df[feature + '_log'] = df[feature + '_log'].replace(-np.inf, np.nan)
    df[feature + '_log'] = df[feature + '_log'].isna().fillna(df[feature + '_log'].mean())
    df = df.drop(columns=[feature])
    return df

def currency_tonumeric(dataframe,feature):
    # Convert from a string to a float
    dataframe[feature] = pd.to_numeric(dataframe[feature].str.replace(',', ''))

def default_nan(dataframe, features_array):
    # Default NaN values with median
    for feature in features_array:
        default_value = dataframe[feature].fillna(dataframe[feature].median())
        dataframe[feature] = dataframe[feature].fillna(default_value).astype(int)

def default_annual_revenue(dataframe):
    # Fix missing values for annual revenue, replace with mean/trimmed mean of the plan size they are on
    plan_list = dataframe.plan[~pd.isnull(dataframe.plan)].unique()

    for plan in plan_list:
        mean = round(dataframe.annual_revenue[dataframe.plan == plan].mean(), 2)
        trimmed_mean = trim_mean(dataframe.annual_revenue[dataframe.plan == plan].values, 0.1)

        if pd.isnull(mean):
            revenue = 0
        else:
            revenue = mean
        dataframe.loc[dataframe.plan==plan, 'annual_revenue'] = dataframe.loc[dataframe.plan==plan, 'annual_revenue'].fillna(revenue)

def features_with_nan(df):
    for feature in df.columns:
        if len(df[df[feature].isna() == True]) > 0:
            print('Feature has NaN values: ' + feature)

def prepare_data(df_raw):
    # Sort data by date
    df_raw = df_raw.sort_values(by='licence_registration_date')

    # Convert annual_revenue from a string to a float
    currency_tonumeric(df_raw, 'annual_revenue')

    # Convert fields to INT and setting any NaNs to the mean of that type
    case_types = ['cases_total','cases_open','cases_closed','cases_age_hours_total','cases_age_hours_average', 'last_login_days']

    default_nan(df_raw, case_types)

    default_annual_revenue(df_raw)

    # 'bin' last login days
    bins = [1, 3, 7, 14, 30, 60]
    group_names = ['day', 'few_days', 'week', 'fortnight', 'month']

    # need to get the mean of the plan size for last_login_days and set each row to that
    #df_raw.last_login_days = df_raw.last_login_days.fillna(np.mean(df_raw.last_login_days))

    last_login_categories = pd.cut(df_raw['last_login_days'], bins, labels=group_names)
    df_raw['last_login_categories'] = pd.cut(df_raw['last_login_days'], bins, labels=group_names)

    # Set default values for NaN values in NPS
    df_raw.nps = df_raw.nps.fillna(np.nanmean(df_raw.nps))

    # 'bin' nps
    # Issue with this is NaN gets given a detractor status which isn't right
    bins = [0,6.9,8.9,10]
    group_names = ['detractor', 'passive', 'promoter']

    df_raw['nps'] = pd.cut(df_raw['nps'], bins, labels=group_names)

    # one-hot encode fields
    dummy_columns = ['customer_account_status', 'last_login_categories', 'plan', 'nps']

    for dummy_column in dummy_columns:
        dummy = pd.get_dummies(df_raw[dummy_column], prefix=dummy_column)
        df_raw = pd.concat([df_raw,dummy], axis=1)
        df_raw = df_raw.drop(columns=dummy_column)


    # This breaks all the date features up into number columns
    # These steps can only be run once then you need to comment them out
    add_datepart(df_raw, 'licence_registration_date')
    add_datepart(df_raw, 'golive_date')

    for feature in ['days_active', 'golive_days', 'cases_age_hours_total', 'annual_revenue']:
        df_raw = logify_feature(df_raw, feature)

    # Drop columns, some of these create "Data Leakage", some are just to test if it has impact when they are taken out
    df_raw = df_raw.drop(columns=['customer_account_status_Good', 'last_login_concern',
                                  'last_login_days', 'account_status', 'changing_platform',
                                  'new_platform', 'licence_status', 'canceldate',
                                  'cancel_details', 'cancel_reason', 'url', 'merchant'])

    # Set NaN to zero
    features = ['churned', 'interactions_total', 'interactions_completed', 'interactions_no_response', 'interactions_no_onboarding', 'interactions_completed_training']

    for feature in features:
        df_raw[feature] = df_raw[feature].fillna(0)

    # Complete the transformation of all data into
    # numbers using proc_df and create training dataframes

    train_cats(df_raw)

    features_with_nan(df_raw)

    return df_raw

