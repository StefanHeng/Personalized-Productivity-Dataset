import os

import pandas as pd
import boto3


if __name__ == '__main__':
    from stefutil import *

    path = os.path.join('cloud-auth', 'mturk iam access key.csv')
    df = pd.read_csv(path)
    auth = df.iloc[0, :].to_dict()
    # mic(auth)

    region_name = 'us-east-1'
    aws_access_key_id = auth['Access key ID']
    aws_secret_access_key = auth['Secret access key']
    PRODUCTION = False
    if PRODUCTION:
        endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'
    else:
        endpoint_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
    client = boto3.client(
        'mturk',
        endpoint_url=endpoint_url,
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    mic(client.get_account_balance()['AvailableBalance'])
