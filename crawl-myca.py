import os
import json
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

        r = requests.post(url=url, data=body)
        mic(r)
        res = json.loads(r.text)
        mic(res)
        fnm = f'{now(for_path=True)}-admin-token.json'
        path_out = os.path.join('auth', 'myca', fnm)
        with open(path_out, 'w') as f:
            json.dump(res, f, indent=4)
    create_token()

    def fetch_data_by_day():
        url = f'{base_url}/js/walker_run'

        token_fnm = '2022-08-02_15-36-01-admin-token.json'
        token_path = os.path.join('auth', 'myca', token_fnm)
        with open(token_path) as f:
            token = json.load(f)['token']

        path = os.path.join('auth', 'myca', 'user-ids.txt')
        with open(path, 'r') as f:
            user_ids = f.read().splitlines()
        user_id = user_ids[0]

        payload = dict(
            name='get_latest_day',
            nd=user_id,
            ctx=dict(
                before_date='2022-08-01',
                # before_date='2022-07-22',
                show_report=1  # this must be 1 or otherwise the response will be empty
            )
        )
        r = requests.post(
            url=url,
            data=json.dumps(payload),  # otherwise, the payload is sent as lists for some unknown reason
            headers={'Authorization': f'token {token}', 'Content-Type': 'application/json'}
        )
        mic(r)
        res = json.loads(r.text)
        mic(res)
    fetch_data_by_day()
