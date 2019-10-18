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

def logify_feature(df, feature, drop=True):
    df[feature + '_log'] = np.log(df[feature])
    df[feature + '_log'] = df[feature + '_log'].replace(-np.inf, np.nan)
    #df[feature + '_log'] = df[feature + '_log'].isna().fillna(df[feature + '_log'].mean())
    if drop:
        df = df.drop(columns=[feature])
    return df

def currency_tonumeric(dataframe,feature):
    # Convert from a string to a float
    dataframe[feature] = pd.to_numeric(dataframe[feature].str.replace(',', ''))

def default_nan(df, features_array):
    # Default NaN values with median
    for feature in features_array:
        default_value = df[feature].fillna(df[feature].median())
        df[feature] = df[feature].fillna(default_value).astype(int)
        df[feature + 'was_nan'] = 1
    return df

def default_annual_revenue(dataframe):
    # Fix missing values for annual revenue, replace with mean/trimmed mean of the plan size they are on
    plan_list = dataframe.plan[~pd.isnull(dataframe.plan)].unique()

    platform_revenue_median = dataframe.annual_revenue.median()

    for plan in plan_list:
        mean = round(dataframe.annual_revenue[dataframe.plan == plan].mean(), 2)
        trimmed_mean = trim_mean(dataframe.annual_revenue[dataframe.plan == plan].values, 0.1)

        if pd.isnull(mean):
            revenue = platform_revenue_median
        else:
            revenue = mean
        dataframe.loc[dataframe.plan==plan, 'annual_revenue'] = dataframe.loc[dataframe.plan == plan, 'annual_revenue'].fillna(revenue)
        dataframe['annual_revenue_was_missing'] = 1

def features_with_nan(df):
    for feature in df.columns:
        if len(df[df[feature].isna() == True]) > 0:
            print('Feature has NaN values: ' + feature)

def preprocess_nps(df):
    # Set default values for NaN values in NPS
    df.nps = df.nps.fillna(-1)

    # 'bin' nps
    # Issue with this is NaN gets given a detractor status which isn't right
    bins = [-2, 0, 6.9, 8.9, 10]
    group_names = ['no_data', 'detractor', 'passive', 'promoter']

    df['nps'] = pd.cut(df['nps'], bins, labels=group_names)
    return df

def preprocess_last_login_days(df):
    # 'bin' last login days
    bins = [1, 3, 7, 14, 30, 60]
    group_names = ['day', 'few_days', 'week', 'fortnight', 'month']

    # need to get the mean of the plan size for last_login_days and set each row to that
    #df_raw.last_login_days = df_raw.last_login_days.fillna(np.mean(df_raw.last_login_days))

    df['last_login_categories'] = pd.cut(df['last_login_days'], bins, labels=group_names)
    return df

def prepare_data(df_raw):
    df_raw = df_raw.sort_values(by='licence_registration_date')

    currency_tonumeric(df_raw, 'annual_revenue')

    df_raw = default_nan(df_raw, ['cases_total', 'cases_open', 'cases_closed',
                                  'cases_age_hours_total', 'cases_age_hours_average',
                                  'last_login_days'])

    default_annual_revenue(df_raw)

    df_raw = preprocess_last_login_days(df_raw)

    df_raw = preprocess_nps(df_raw)

    # one-hot encode fields
    #dummy_columns = ['customer_account_status', 'last_login_categories', 'plan', 'nps']
    dummy_columns = ['customer_account_status', 'plan', 'nps']

    for dummy_column in dummy_columns:
        dummy = pd.get_dummies(df_raw[dummy_column], prefix=dummy_column)
        df_raw = pd.concat([df_raw,dummy], axis=1)
        df_raw = df_raw.drop(columns=dummy_column)


    # This breaks all the date features up into number columns
    # These steps can only be run once then you need to comment them out
    add_datepart(df_raw, 'licence_registration_date')
    add_datepart(df_raw, 'golive_date')

    # Disabled because we dont necissarily need normal distribution for Random Forest models
    #for feature in ['days_active', 'golive_days', 'cases_age_hours_total', 'annual_revenue']:
    #    df_raw = logify_feature(df_raw, feature)

    # Drop columns, some of these create "Data Leakage", some are just to test if it has impact when they are taken out
    df_raw = df_raw.drop(columns=['customer_account_status_Good', 'last_login_concern',
                                  'last_login_days', 'account_status', 'changing_platform',
                                  'new_platform', 'licence_status', 'canceldate',
                                  'cancel_details', 'cancel_reason', 'url', 'merchant',
                                  'total_churn_concern_cases_age', 'username'])

    # Set any remaining NaNs to the features median
    for feature in df_raw.select_dtypes(include=['float64', 'int64']).columns:
        median = df_raw[feature].median()
        #print(f"{feature}: {median}")
        df_raw[feature] = df_raw[feature].fillna(df_raw[feature].median())
        df_raw[feature + '_was_nan'] = 1

    # Complete the transformation of all data into
    # numbers using proc_df and create training dataframes
    train_cats(df_raw)

    features_with_nan(df_raw)

    return df_raw

