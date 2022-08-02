import os
import requests

import pandas as pd

from stefutil import *


if __name__ == '__main__':
    base_url = 'https://jasecidev.lifelogify.com'

    def create_token():
        url = f'{base_url}/user/token/'
        path = os.path.join('auth', 'myca', 'admin-credential.csv')

        df = pd.read_csv(path)
        auth = df.iloc[0, :].to_dict()
        body = dict(email=auth['username'], password=auth['password'])
        headers = {'Content-Type': 'application/json'}  # adding this makes parse error?

        r = requests.post(
            url,
            data=body,
            # headers=headers
        )
        mic(r)
        mic(r.text)
    # create_token()

    def fetch_data_by_day():
        url = f'{base_url}/js/walker_run'

        path = os.path.join('auth', 'myca', 'user-ids.txt')
        with open(path, 'r') as f:
            user_ids = f.read().splitlines()
            mic(user_ids)
        user_id = user_ids[0]

        body = dict(
            name="get_latest_day",
            nd=user_id,
            ctx=dict(
                before_date="2022-08-01",
                show_report=1  # this must be 1 or otherwise the response will be empty
            )
        )
        r = requests.post(
            url,
            data=body
        )
        mic(r)
        mic(r.text)
    fetch_data_by_day()
