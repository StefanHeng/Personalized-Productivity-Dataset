import os
import json
import datetime
import requests
from typing import List, Tuple, Dict

import pandas as pd
from tqdm.auto import tqdm

from stefutil import *


class ApiCaller:
    """
    Makes API call to return all entries added for a given user on a given day
    """
    base_url = 'https://jasecidev.lifelogify.com'
    init_url = f'{base_url}/user/token/'
    call_url = f'{base_url}/js/walker_run'

    def __init__(self, credential_fnm: str = 'admin-credential.csv', save_token: bool = False, verbose: bool = True):
        self.logger = get_logger(self.__class__.__qualname__)
        # TODO: without this, logging message is duplicated for unknown reason in this project only
        self.logger.propagate = False
        self.verbose = verbose

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
            if self.verbose:
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
        if self.verbose:
            self.logger.info(f'Making fetch data call with {logi(args)}... ')
        t_strt = datetime.datetime.now()
        res = requests.post(**args)
        t = fmt_delta(datetime.datetime.now() - t_strt)
        if self.verbose:
            self.logger.info(f'Got response in {logi(t)} with status {logi(res.status_code)}')
        assert res.status_code == 200
        res = json.loads(res.text)
        assert res['success']
        return res['report']


def get_user_ids(path: str = os.path.join('auth', 'myca', 'user-ids.txt')) -> List[str]:
    with open(path, 'r') as f:
        # keeping the prefix still works, but not friendly to file system
        return [i.removeprefix('urn:uuid:') for i in f.read().splitlines()]


class DataWriter:
    """
    Writes raw action entries per day returned from myca API calls for a given user
    """
    def __init__(self, output_path: str = os.path.join('myca-dataset', f'raw, {now(for_path=True)}')):
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)

        self.ac = ApiCaller(verbose=False)

    @staticmethod
    def _map_entry(f1: str, entry: Dict) -> Dict:
        """
        Store every possible field returned from the API call
        """
        #     assert uid == user_id and user_id == entry['jid']
        #     return dict(
        #         context_name=entry['context']['name'],
        #         context_note=entry['context']['note'],
        #         j_timestamp=entry['j_timestamp'],
        #         j_type=entry['j_type'],
        #         kind=entry['kind'],
        #         name=entry['name']
        #     )
        d = dict(field1=f1)
        d_cont = entry.pop('context')
        d.update({f'context.{k}': v for k, v in d_cont.items()})
        d.update(entry)
        return d

    def get_single(self, user_id: str, before_date: str, write_csv: bool = False) -> pd.DataFrame:
        entries = self.ac(user_id=user_id, before_date=before_date)
        df = pd.DataFrame([self._map_entry(*e) for e in entries])
        if write_csv:
            self.write_single(user_id=user_id, date=before_date, df=df)
        return df

    def write_single(self, user_id: str = None, date: str = None, df: pd.DataFrame = None, group_by_user: bool = True):
        if df is None:
            df = self.get_single(user_id=user_id, before_date=date)
        if group_by_user:
            path = os.path.join(self.output_path, user_id)
            os.makedirs(path, exist_ok=True)
            path = os.path.join(path, f'{date}.csv')
        else:
            path = os.path.join(self.output_path, f'{user_id}-{date}.csv')
        df.to_csv(path, index=False)

    def get_all(self, user_id: str, start_date: str, end_date: str, write_csv: bool = False) -> Dict[str, pd.DataFrame]:
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        dt2df: Dict[str, pd.DataFrame] = dict()
        it = tqdm(dates, desc='Processing', unit='date')
        n = 0
        for d in it:
            d = d.strftime('%Y-%m-%d')
            it.set_postfix(dict(n=n, date_q=d))
            df = self.get_single(user_id=user_id, before_date=d)
            if not df.empty:
                day = df.loc[0, 'context.day']
                day = datetime.datetime.strptime(day, '%Y-%m-%dT%H:%M:%S')
                assert day.hour == 0 and day.minute == 0 and day.second == 0
                day = day.strftime('%Y-%m-%d')
                it.set_postfix(dict(n=n, date_q=d, date_ret=day))
                # Ensure no duplicates
                if day not in dt2df:
                    dt2df[day] = df
                    n += 1
                else:
                    assert df.equals(dt2df[day])
        if write_csv:
            for d, df in dt2df.items():
                self.write_single(user_id=user_id, date=d, df=df)
        return dt2df


if __name__ == '__main__':
    def check_call():
        ac = ApiCaller()
        user_id = get_user_ids()[0]
        mic(user_id)
        entries = ac(user_id=user_id, before_date='2022-08-01')
        mic(entries)
    # check_call()

    def fetch_data_by_day():
        user_id = get_user_ids()[0]
        before_date = '2022-08-01'

        dw = DataWriter()
        df = dw.get_single(user_id=user_id, before_date=before_date)
        mic(df)
    # fetch_data_by_day()

    def fetch_all():
        user_id = get_user_ids()[0]
        dw = DataWriter()
        # st = '2021-01-01'
        st = '2022-07-15'
        dw.get_all(user_id=user_id, start_date=st, end_date='2022-08-01', write_csv=True)
    fetch_all()
