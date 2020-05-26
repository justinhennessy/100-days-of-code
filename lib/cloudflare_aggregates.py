import datetime
from io import StringIO
import boto3
import pandas as pd
import sys
import os

def generate_file_name(processing_date, file_format):
    processing_date = datetime.datetime.strptime(DATE, "%d-%m-%Y")
    filename = str(processing_date.year) + '-' + str(processing_date.strftime("%B").lower()) + '.' + str(file_format)
    return filename

def directory_exists(bucket, key):
    response = s3_client.list_objects_v2(
        Bucket=bucket,
        Prefix=key,
    )
    number_of_objects = response['KeyCount']
    
    return True if number_of_objects > 0 else False

def create_cloudflare_directory_structure(bucket):
    s3_client       = boto3.client('s3')
    directories     = [
        s3_base_path,
        s3_base_path + 'plan_aggregates/csv/',
        s3_base_path + 'plan_aggregates/json/',
        s3_base_path + 'merchant_aggregates/csv/',
        s3_base_path + 'merchant_aggregates/json/',
        s3_base_path + 'total_aggregates/csv/',
        s3_base_path + 'total_aggregates/json/',
        s3_base_path + 'high_utilisation/csv/',
        s3_base_path + 'high_utilisation/json/'
        
    ]
    
    for directory in directories:
        if not directory_exists(bucket, directory):
            print('Creating directory: ' + directory)
            s3_client.put_object(Bucket=bucket, Key=directory)
            
def load_s3_object(bucket, key):
    s3 = boto3.resource('s3')
    obj = s3.Object(bucket, key)
    body = obj.get()['Body']
    csv_string = body.read().decode('utf-8')
    dataframe = pd.read_csv(StringIO(csv_string), parse_dates=True)
    return dataframe

def create_active_subscriptions_dataframe(bucket, file_path):
    # Load in and clean up active subsciptions data using Lake Neto
    dataframe = load_s3_object(bucket, file_path)
    dataframe['merchant_id'] = dataframe['reference']
    dataframe = dataframe.drop(columns=['customerId','revenueLimit', 'status', 'planName', 'planFrequency.interval'])
    dataframe = dataframe.set_index('reference')
    dataframe = dataframe[~dataframe.index.isna()]
    return dataframe

def create_cloudflare_dataframe(bucket, file_path):
    # Load in cloudflare month aggregate data into a dataframe
    dataframe = load_s3_object(bucket, file_path)
    dataframe = dataframe.drop(columns=['requests'])
    return dataframe

def create_domain_id_map_dataframe(bucket, file_path):
    # Load in the domain to N Number mapping data into a dataframe
    dataframe = load_s3_object(bucket, file_path)
    dataframe = dataframe.drop(columns=['blah'])
    dataframe = dataframe.set_index('id')
    return dataframe

def create_totals_dataframe(cloudflare_data):
    dataframe = cloudflare_data._get_numeric_data().agg(sum).to_frame()
    dataframe = dataframe.rename(columns={0: 'transfer_bytes'})
    dataframe = dataframe.reset_index(drop=True)
    return dataframe

def generate_plan_aggregates(cloudflare_dataframe):
    # Generate an average transfer for each plan type
    dataframe = cloudflare_dataframe.groupby('planCode').agg('mean')

    dataframe = dataframe.rename(columns={'transfer': 'transfer_bytes_mean'})
    
    for row in dataframe.itertuples():
        # Calculate percentiles
        p25 = cloudflare_dataframe[cloudflare_dataframe['planCode'] == row.Index].transfer.quantile([.25]).values[0]
        p50 = cloudflare_dataframe[cloudflare_dataframe['planCode'] == row.Index].transfer.quantile([.50]).values[0]
        p75 = cloudflare_dataframe[cloudflare_dataframe['planCode'] == row.Index].transfer.quantile([.75]).values[0]
        transfer_iqr = p75 - p25
        transfer_iqr_max = p75 + 1.5 * transfer_iqr # This is refered to a Terkey's fence, MAX

        dataframe.at[row.Index, 'transfer_bytes_p25'] = p25
        dataframe.at[row.Index, 'transfer_bytes_p50'] = p50    
        dataframe.at[row.Index, 'transfer_bytes_p75'] = p75
        dataframe.at[row.Index, 'transfer_bytes_iqr'] = transfer_iqr
        dataframe.at[row.Index, 'transfer_bytes_iqr_max'] = transfer_iqr_max
        
    return dataframe
    
def generate_merchant_aggregates(cloudflare_dataframe, plan_dataframe):
    # Iterate over the df_final dataframe and calculate transfer delta for each Merchant
    for row in cloudflare_dataframe.itertuples():
        plan = row.planCode
        plan_median = plan_dataframe.loc[plan].transfer_bytes_p50
        delta = row.transfer - plan_median
        delta_gb = delta/1000000000
        is_outlier = row.transfer > plan_dataframe.loc[plan].transfer_bytes_iqr_max

        # Sets the value of the delta field for each row as it iterates
        cloudflare_dataframe.at[row.Index, 'delta_bytes']        = delta
        cloudflare_dataframe.at[row.Index, 'delta_gb']           = delta_gb
        cloudflare_dataframe.at[row.Index, 'is_outliner']        = is_outlier
        cloudflare_dataframe.at[row.Index, 'utilisation_rating'] = 0
    
        if is_outlier == True:
            cloudflare_dataframe.at[row.Index, 'utilisation_rating'] = 1
            if delta_gb > 200:
                cloudflare_dataframe.at[row.Index, 'utilisation_rating'] = 2

    cloudflare_dataframe['utilisation_rating'] = pd.to_numeric(cloudflare_dataframe['utilisation_rating'], downcast='integer')
    return cloudflare_dataframe

def create_merchant_dataframe(df_cloudflare, df_active, df_idmap):
    # Join the active sites dataframe with the idmap dataframe using the N number as they index for both dataframes
    df_activesites = df_active.join(df_idmap, how='left')

    # Duplicating the domainprimary column before we use it as an index as we will need to main later
    df_activesites['domain'] = df_activesites['domainprimary']

    # Joins the activesites dataframe with the cloudflare dataframe using the site domain as the index
    df_activesites = df_activesites.set_index('domainprimary').join(df_cloudflare.set_index('domain'), how='left')

    # Filter out rows that have NaN in transfer, which means they didn't match in the join
    df_final = df_activesites[~df_activesites['transfer'].isna()]

    # Reset dataframe index to default (ie sequential numbers) as we no longer need it to be domain
    df_final = df_final.reset_index(drop=True)
    
    return df_final
    
def render_s3_path(bucket, path, file_format):
    s3_path = 's3://' + bucket + '/' + s3_base_path + path + '/' + file_format + '/'
    return s3_path

def upload_file(dataframe, path):
    dataframe.to_csv(render_s3_path(bucket, path, 'csv') + generate_file_name(DATE, 'csv'))
