"""
Getting raw dataset via API calls & storing to file system
"""


import os
import json
import datetime
import requests
from os.path import join as os_join
from typing import List, Tuple, Dict
from dataclasses import dataclass

import pandas as pd
from tqdm.auto import tqdm

from stefutil import *
from myca.util import *


__all__ = ['ApiCaller', 'WriteOutput', 'DataWriter']


class ApiCaller:
    """
    Makes API call to return all entries added for a given user on a given day
    """
    auth_path = os_join(u.proj_path, 'auth', 'myca')

    def __init__(
            self, credential_fnm: str = 'admin-credential.csv', token_fnm: str = None, verbose: bool = True,
            env: str = 'dev'
    ):
        ca.check_mismatch('API Env', env, ['dev', 'prod'])
        self.env = env
        if env == 'dev':
            self.base_url = 'https://jasecidev.lifelogify.com'
        else:
            self.base_url = 'https://sandbox.myca.ai'
        self.init_url = f'{self.base_url}/user/token/'
        self.call_url = f'{self.base_url}/js/walker_run'

        self.logger = get_logger(self.__class__.__qualname__)
        self.verbose = verbose

        self.token_fnm = token_fnm
        token_path = os_join(ApiCaller.auth_path, f'{token_fnm}.json')
        if token_fnm and not os.path.exists(token_path):
            credential_path = os_join(ApiCaller.auth_path, credential_fnm)
            df = pd.read_csv(credential_path)
            auth = df.iloc[0, :].to_dict()
            payload = dict(email=auth['username'], password=auth['password'])

            res = requests.post(url=self.init_url, data=payload)
            res = json.loads(res.text)
            with open(token_path, 'w') as f:
                json.dump(res, f, indent=4)
            if self.verbose:
                self.logger.info(f'Admin token saved to {logi(token_path)}')

    def __call__(self, user_id: str, before_date: str, token_fnm: str = None) -> List[Tuple[str, Dict]]:
        token_path = os_join(ApiCaller.auth_path, f'{token_fnm or self.token_fnm}.json')
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
        args = dict(url=self.call_url, headers=headers, data=json.dumps(payload))
        if self.verbose:
            self.logger.info(f'Making fetch dataset call with {logi(args)}... ')

        res = None
        while not res or res.status_code != 200:  # Retry if code is 503
            t_strt = datetime.datetime.now()
            res = requests.post(**args)
            t = fmt_delta(datetime.datetime.now() - t_strt)
            if self.verbose:
                self.logger.info(f'Got response in {logi(t)} with status {logi(res.status_code)}')
        if res.status_code == 200:
            res = json.loads(res.text)
        else:
            raise ValueError(f'API call failed with {logi(res)}')
        if res['success']:
            return res['report']
        else:
            print(res['stack_trace'])
            raise ValueError(f'API call failed with {logi(res)}')


@dataclass
class WriteOutput:
    text: List[Tuple[str, Dict]] = None
    table: pd.DataFrame = None


class DataWriter:
    """
    Writes raw action entries per day returned from myca API calls for a given user
    """
    def __init__(
            self, output_path: str = os_join(u.dset_path, f'raw, {now(for_path=True)}'), save_raw: bool = False,
            caller_args: Dict = None
    ):
        self.logger = get_logger(self.__class__.__qualname__)

        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)

        self.ac = ApiCaller(verbose=False, **caller_args)
        self.save_raw = save_raw

    @staticmethod
    def _map_entry(f1: str, entry: Dict) -> Dict:
        """
        Store every possible field returned from the API call
        """
        d = dict(field1=f1)
        # note the 1st element is unique, with different set of keys compared to subsequent, actual action entries
        d.update({f'context.{k}': v for k, v in entry['context'].items()})
        d.update({k: v for k, v in entry.items() if k != 'context'})
        return d

    def get_single(self, user_id: str, before_date: str, save: bool = False) -> WriteOutput:
        entries = self.ac(user_id=user_id, before_date=before_date)
        df = pd.DataFrame([self._map_entry(*e) for e in entries])
        order = sorted(list(df.columns))
        order.insert(0, order.pop(order.index('field1')))
        df = df[order]
        if save:
            self.write_single(user_id=user_id, date=before_date, out=df)
        return WriteOutput(text=entries, table=df)

    def write_single(self, user_id: str = None, date: str = None, out: WriteOutput = None, group_by_user: bool = True):
        if out is None:
            out = self.get_single(user_id=user_id, before_date=date)
        if group_by_user:
            path = os_join(self.output_path, user_id)
            os.makedirs(path, exist_ok=True)
            path = os_join(path, f'{date}.csv')
        else:
            path = os_join(self.output_path, f'{user_id}-{date}.csv')
        raw, df = out.text, out.table
        df.to_csv(path, index=False)
        if self.save_raw:
            path = path.removesuffix('.csv') + '.json'
            with open(path, 'w') as f:
                json.dump(raw, f, indent=4)

    def get_all(self, user_id: str, start_date: str, end_date: str, save: bool = False) -> Dict[str, WriteOutput]:
        d_log = dict(user_id=user_id, start_date=start_date, end_date=end_date)
        self.logger.info(f'Getting raw dataset with {logi(d_log)}... ')
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        dt2df: Dict[str, WriteOutput] = dict()
        it = tqdm(dates, desc='Processing', unit='date')
        n = 0
        for d in it:
            d = d.strftime('%Y-%m-%d')
            it.set_postfix(dict(n=n, date_q=d))
            out = self.get_single(user_id=user_id, before_date=d)
            df = out.table
            if not df.empty:
                day = df.loc[0, 'context.day']
                day = datetime.datetime.strptime(day, '%Y-%m-%dT%H:%M:%S')
                assert day.hour == 0 and day.minute == 0 and day.second == 0
                day = day.strftime('%Y-%m-%d')
                it.set_postfix(dict(n=n, date_q=d, date_ret=day))
                # Ensure no duplicates
                if day not in dt2df:
                    dt2df[day] = out
                    n += 1
                else:
                    assert df.equals(dt2df[day].table)  # by `get_single` construction, `text` is also the same
        if save:
            for d, o in dt2df.items():
                self.write_single(user_id=user_id, date=d, out=o)
        return dt2df


if __name__ == '__main__':
    from myca.dataset.util import *

    def check_call():
        e = 'prod'
        tok = '2022-08-08_21-54-23-prod-admin-token'
        # tok = f'{now(for_path=True)}-{e}-admin-token'
        ac = ApiCaller(env=e, token_fnm=tok)
        user_id = get_user_ids(split=e)[0]
        entries = ac(user_id=user_id, before_date='2022-07-29', token_fnm=tok)
        mic(entries)
    # check_call()

    def fetch_data_by_day():
        user_id = get_user_ids()[0]
        before_date = '2022-08-01'

        dw = DataWriter()
        df = dw.get_single(user_id=user_id, before_date=before_date)
        mic(df)
    # fetch_data_by_day()

    def write_all():
        # e = 'dev'
        e = 'prod'

        if e == 'dev':
            tok_fnm = '2022-08-08_21-53-38-dev-admin-token'
            dw = DataWriter(caller_args=dict(env=e, token_fnm=tok_fnm), save_raw=True)
            user_id = get_user_ids()[0]
            dw.get_all(user_id=user_id, start_date='2022-07-15', end_date='2022-08-01', save=True)
        else:
            tok_fnm = '2022-08-08_21-54-23-prod-admin-token'
            dw = DataWriter(caller_args=dict(env=e, token_fnm=tok_fnm), save_raw=True)
            uids = get_user_ids(split=e)[2:]
            # st = '2020-09-01'  # This got error from Myca API
            st = '2020-10-01'
            # st = '2021-01-01'
            # st = '2021-10-28'
            for i in uids:
                dw.get_all(user_id=i, start_date=st, end_date='2022-08-01', save=True)
    # write_all()
