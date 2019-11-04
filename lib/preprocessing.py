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

def random_replace_value(df, feature):
    median = df[feature].median()
    mean = df[feature].mean()
    trimmed_mean = trim_mean(df[feature].values, 0.1)

    random_choice = random.randint(1, 3)

    if random_choice == 1:
        value = mean
    elif random_choice == 2:
        value = trimmed_mean
    else:
        value = median

    return value

def fix_days_active(df):
    feature = 'days_active'
    df[feature + '_modified'] = 0

    for i in df[(df['licence_registration_date'] == '11/25/2013') & (df['licence_status'] != "CANCELLED")].index:
        df[feature][i] = random_replace_value(df, feature)
        df.loc[i, feature + '_modified'] = 1
    return df

def currency_tonumeric(dataframe, feature):
    # Convert from a string to a float
    dataframe[feature] = pd.to_numeric(dataframe[feature].str.replace(',', ''))

def default_nan(df, features_array):
    # Default NaN values with median
    for feature in features_array:
        default_value = df[feature].fillna(df[feature].median())
        df[feature] = df[feature].fillna(default_value).astype(int)
        df[feature + '_was_nan'] = 1
    return df

def get_plan_list(df):
   return df.plan[~pd.isnull(df.plan)].unique()

def get_plan_mean(df, plan, feature):
    return abs(round(df[feature][df.plan == plan].mean(), 2))

def get_plan_trimmed_mean(df, plan, feature):
    return trim_mean(df[feature][df.plan == plan].values, 0.1)

def get_platform_median(df, feature):
    return df[feature].median()

def get_platform_mean(df, feature):
    return df[feature].mean()

def default_last_login_days(df):
    platform_last_login_days_median = get_platform_median(df, 'last_login_days')
    print(f"  Platform last_login_days median: {platform_last_login_days_median}")

    platform_last_login_days_mean = get_platform_mean(df, 'last_login_days')
    print(f"  Platform last_login_days mean: {platform_last_login_days_mean}")

    plan_list = get_plan_list(df)

    #df['last_login_days_was_missing'] = 0
    #df.last_login_days_was_missing.loc[df.last_login_days.isna()] = 1

    # Replace NaN last_login_days values with an appropriate value based on plan
    for plan in plan_list:
        mean = get_plan_mean(df, plan, 'last_login_days')

        if pd.isnull(mean):
            last_login_days = platform_last_login_days_median
        else:
            last_login_days = mean

        df.loc[df.plan == plan, 'last_login_days'] = df.loc[df.plan == plan, 'last_login_days'].fillna(last_login_days)

    print("  List records that still have NaN in them ...")
    print(f"  {len(df.loc[df.last_login_days.isna()])}")
    return df

def set_no_plan(df):
    df['plan_not_set'] = 0

    for i in df[df.plan.isna() == True].index:
        df['plan'][i] = "no_plan"
        df['plan_not_set'][i] = 1

    return df

def default_annual_revenue(df):
    # Fix missing values for annual revenue, replace with mean/trimmed mean of the plan size they are on

    platform_revenue_median = get_platform_median(df, 'annual_revenue')
    print(f" Platform revenue median: {platform_revenue_median}")

    df['annual_revenue_was_missing'] = 0
    df.annual_revenue_was_missing.loc[df.annual_revenue.isna()] = 1

    plan_list = get_plan_list(df)

    # Replace NaN annual_revenue values with an appropriate value based on plan
    for plan in plan_list:
        mean = get_plan_mean(df, plan, 'annual_revenue')
        trimmed_mean = get_plan_trimmed_mean(df, plan, 'annual_revenue')

        if pd.isnull(mean):
            revenue = platform_revenue_median
        else:
            revenue = mean

        df.loc[df.plan == plan, 'annual_revenue'] = \
            df.loc[df.plan == plan, 'annual_revenue'].fillna(revenue)
        df.annual_revenue.loc[(df.annual_revenue == 0) & (df.plan == plan)] = revenue

    # Output any rows that have NaN for plan or annual_revenue
    if len(df[df.plan.isna() == True]) > 0:
      print("Output customers with no plan set ...")
      print(df.username[df.plan.isna() == True])

    if len(df[df['annual_revenue'].isna() == True]) > 0:
        print("Output records with a zero revenue ...")
        print(df.username[df['annual_revenue'].isna() == True])

    return df

def features_with_nan(df):
    for feature in df.columns:
        if len(df[df[feature].isna() == True]) > 0:
            print(f"Feature has NaN values: {feature}, {len(df[df[feature].isna() == True])}")

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
    bins = [-1, 2, 4, 7, 15, 30, df['last_login_days'].max()]
    group_names = ['day', 'few_days', 'week', 'fortnight', 'month', 'month+']

    # need to get the mean of the plan size for last_login_days and set each row to that
    #df_raw.last_login_days = df_raw.last_login_days.fillna(np.mean(df_raw.last_login_days))

    df['last_login_categories'] = pd.cut(df['last_login_days'], bins, labels=group_names)
    print(df['last_login_categories'].value_counts())
    return df

def prepare_data(df_raw):
    print("Sorting dataframe by licence_registration_date ...")
    df_raw = df_raw.sort_values(by='licence_registration_date')

    #print("Converting annual_revenue to a number ...")
    #currency_tonumeric(df_raw, 'annual_revenue')

    # Set plan type to no_plan if it is missing
    df_raw = set_no_plan(df_raw)

    print("Defaulting last_login_days for missing values to the plans mean or platform medeian ...")
    df_raw = default_last_login_days(df_raw)

    print("Defaulting all NaN values with median ...")
    default_fields = ['cases_total', 'cases_open', 'cases_closed',
                                  'cases_age_hours_total', 'cases_age_hours_average']
    df_raw = default_nan(df_raw, default_fields)

    #print("Defaulting annual_revenue for missing values to the plans mean or platform medeian ...")
    #df_raw = default_annual_revenue(df_raw)

    print("Binning last_login_days")
    df_raw = preprocess_last_login_days(df_raw)

    print("Binning NPS data")
    df_raw = preprocess_nps(df_raw)

    print("Fixing days_active abnormal values ...")
    df_raw = fix_days_active(df_raw)

    # one-hot encode fields
    dummy_columns = ['customer_account_status', 'plan', 'nps', 'last_login_categories']

    for dummy_column in dummy_columns:
        print(f"One-hot encoding {dummy_column}")
        dummy = pd.get_dummies(df_raw[dummy_column], prefix=dummy_column)
        df_raw = pd.concat([df_raw,dummy], axis=1)
        df_raw = df_raw.drop(columns=dummy_column)

    print("Preprocessing dates ...")
    # This breaks all the date features up into number columns
    # These steps can only be run once then you need to comment them out
    add_datepart(df_raw, 'licence_registration_date')
    #add_datepart(df_raw, 'golive_date')

    # Disabled because we dont necissarily need normal distribution for Random Forest models
    #for feature in ['days_active', 'golive_days', 'cases_age_hours_total', 'annual_revenue']:
    #    df_raw = logify_feature(df_raw, feature)

    # Drop columns, some of these create "Data Leakage", some are just to test if it has impact when they are taken out
    for feature in ['customer_account_status_Good', 'last_login_days', 'account_status', \
                                  'canceldate', 'url', \
                                  'total_churn_concern_cases_age']:
        print(f"Dropping feature {feature} ...")
        df_raw = df_raw.drop(columns=[feature])

    for regex_filter in ['interactions', 'churn_concern']:
        print(f"Dropping features with '{regex_filter}' in their name ...")
        df_raw = df_raw[df_raw.columns.drop(list(df_raw.filter(regex=regex_filter)))]

    print("Convert categorical features into numbers ...")
    # Complete the transformation of all data into
    # numbers using proc_df and create training dataframes
    train_cats(df_raw)
    #df_raw.last_login_categories.cat.set_categories(['day', 'few_days', 'week', 'fortnight', 'month', 'month+'], ordered=True, inplace=True)

    print("List any features that still have NaN values ...")
    features_with_nan(df_raw)

    return df_raw

