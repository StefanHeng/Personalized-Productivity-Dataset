import os
import json
import datetime
import requests
from typing import List, Tuple, Dict

import pandas as pd

from stefutil import *


class ApiCaller:
    """
    Makes API call to return all entries added for a given user on a given day
    """
    base_url = 'https://jasecidev.lifelogify.com'
    init_url = f'{base_url}/user/token/'
    call_url = f'{base_url}/js/walker_run'

    def __init__(self, credential_fnm: str = 'admin-credential.csv', save_token: bool = False):
        self.logger = get_logger(self.__class__.__qualname__)
        # TODO: without this, logging message is duplicated for unknown reason in this project only
        self.logger.propagate = False

        if save_token:
            credential_path = os.path.join('auth', 'myca', credential_fnm)
            df = pd.read_csv(credential_path)
            auth = df.iloc[0, :].to_dict()
            payload = dict(email=auth['username'], password=auth['password'])

            res = requests.post(url=ApiCaller.init_url, data=payload)
            res = json.loads(res.text)
            fnm = f'{now(for_path=True)}-admin-token.json'
            path_out = os.path.join('auth', 'myca', fnm)
            with open(path_out, 'w') as f:
                json.dump(res, f, indent=4)
            self.logger.info(f'Admin token saved to {logi(path_out)}')

    def __call__(
            self, user_id: str, before_date: str, token_fnm: str = '2022-08-02_15-36-01-admin-token.json'
    ) -> List[Tuple[str, Dict]]:
        token_path = os.path.join('auth', 'myca', token_fnm)
        with open(token_path) as f:
            token = json.load(f)['token']

        headers = {
            'Authorization': f'token {token}',
            'Content-Type': 'application/json'  # otherwise, the payload is sent as lists for some unknown reason
        }
        payload = dict(
            name='get_latest_day',
            nd=user_id,
            ctx=dict(
                before_date=before_date,
                show_report=1  # this must be 1 or otherwise the response will be empty
            )
        )
        args = dict(url=ApiCaller.call_url, headers=headers, data=json.dumps(payload))
        self.logger.info(f'Making fetch data call with {logi(args)}... ')
        t_strt = datetime.datetime.now()
        res = requests.post(**args)
        t = fmt_delta(datetime.datetime.now() - t_strt)

        self.logger.info(f'Got response in {logi(t)} with status {logi(res.status_code)}')
        assert res.status_code == 200
        res = json.loads(res.text)
        assert res['success']
        return res['report']


if __name__ == '__main__':
    ac = ApiCaller()

    dset_out_path = os.path.join('myca-dataset', 'raw')
    os.makedirs(dset_out_path, exist_ok=True)

    def fetch_data_by_day(before_date: str):
        path = os.path.join('auth', 'myca', 'user-ids.txt')
        with open(path, 'r') as f:
            user_ids = f.read().splitlines()
        user_id = user_ids[0]
        user_id = user_id.removeprefix('urn:uuid:')
        mic(user_id)

        entries = ac(user_id=user_id, before_date=before_date)

        # def map_entry(uid: str, entry: Dict) -> Dict:
        #     assert uid == user_id and user_id == entry['jid']
        #     return dict(
        #         context_name=entry['context']['name'],
        #         context_note=entry['context']['note'],
        #         j_timestamp=entry['j_timestamp'],
        #         j_type=entry['j_type'],
        #         kind=entry['kind'],
        #         name=entry['name']
        #     )

        def map_entry(f1: str, entry: Dict) -> Dict:
            """
            Store every possible field returned from the API call
            """
            d = dict(field1=f1)
            d_cont = entry.pop('context')
            d.update({f'context.{k}': v for k, v in d_cont.items()})
            d.update(entry)
            return d

        df = pd.DataFrame([map_entry(*e) for e in entries])
        mic(df)
        df.to_csv(os.path.join(dset_out_path, f'{user_id}-{before_date}.csv'), index=False)
    dt = '2022-08-01'
    fetch_data_by_day(before_date=dt)
