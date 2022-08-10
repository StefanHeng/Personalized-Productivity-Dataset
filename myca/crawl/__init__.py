import os
import math
import json
import glob
import datetime
import requests
from os.path import join as os_join
from typing import List, Tuple, Dict, Union, Any
from dataclasses import dataclass

import pandas as pd
from tqdm.auto import tqdm

from stefutil import *
from myca.util import *


__all__ = ['ApiCaller', 'WriteOutput', 'DataWriter', 'DataCleaner']


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
        # TODO: without this, logging message is duplicated for unknown reason in this project only
        self.logger.propagate = False
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
            self.logger.info(f'Making fetch data call with {logi(args)}... ')

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


def get_user_ids(path: str = os_join(ApiCaller.auth_path, 'user-ids.json'), split: str = 'dev') -> List[str]:
    with open(path, 'r') as f:
        user_ids = json.load(f)[split]
        # keeping the prefix still works, but not friendly to file system
        return [i.removeprefix('urn:uuid:') for i in user_ids]


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
        self.logger.propagate = False

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
        self.logger.info(f'Getting raw data with {logi(d_log)}... ')
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


def _path2root(graph: Dict, start, target, curr_path: List) -> List:
    children = graph[start]
    if children:
        if target in children:
            curr_path.append(target)
            return curr_path
        else:
            for c in children:
                if c not in curr_path:  # Not necessary since tree
                    curr_path.append(c)
                    path = _path2root(graph, c, target, curr_path)
                    if path:
                        return path
                    curr_path.pop()


def path2root(graph: Dict, root, target) -> List:
    return _path2root(graph, root, target, [root])  # inclusive start & end


def readable_tree(graph: Dict[str, List[str]], root: str, parent_prefix: str = None) -> Union[str, Tuple[str, List[Any]]]:
    """
    :param graph: Adjacency list of a graph/tree
    :param root: Root node
    :param parent_prefix: Needed for precision to differentiate inner & leaf nodes
        Counter example: 2 nodes, A, B are neighbors of each other, A is leaf node, B is inner node
        But they would appear as if A has a single child B
        This is because Tuple is rendered as List in json
    :return: nested binary tuples of (name, children)
    """
    children = graph[root]
    is_leaf = (not children) or len(children) == 0
    if is_leaf:
        return root
    else:
        p = f'{parent_prefix}_{root}' if parent_prefix else root
        return p, [readable_tree(graph, c, parent_prefix=parent_prefix) for c in children]


class DataCleaner:
    """
    Clean up the raw data (see `DataWriter`) from a given date into our format
    """
    def __init__(
            self, dataset_path: str, output_path: str = os_join(u.dset_path, f'cleaned, {now(for_path=True)}'),
            verbose: bool = True
    ):
        self.logger = get_logger(self.__class__.__qualname__)
        self.logger.propagate = False
        self.verbose = verbose

        self.dataset_path = dataset_path
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        # folder name is user id, see `DataWriter`
        self.user_ids = [stem(f) for f in sorted(glob.iglob(os_join(dataset_path, '*')))]
        self.uid2dt: Dict[str, List[str]] = {
            uid: [stem(f) for f in sorted(glob.glob(os_join(dataset_path, uid, '*.csv')))]
            for uid in self.user_ids
        }

    @staticmethod
    def _clean_single_entry(entry: pd.Series) -> pd.Series:
        return pd.Series(dict(
            id=entry['jid'],  # id of the action entry
            text=entry['context.name'],  # actual text for the action entry
            note=entry['context.note'],
            link=entry['context.links'],
            creation_time=entry['j_timestamp'],
            type=entry['context.wtype'],  # UI type of the entry
            parent_id=entry['field1']  # by API call design
        ))

    @staticmethod
    def _cleaned_df2graph(df: pd.DataFrame):
        """
        Build the hierarchy for entry groups, as graph/tree represented with adjacency list, i.e. node => children
        Action items that definitely will not have any children will have value of None, instead of empty list

        .. note:: iteration order is preserved w.r.t the dataframe
        """
        root_id = df.loc[0, 'id']
        graph: Dict[str, List[str]] = {root_id: []}
        for _, row in df.iloc[1:].iterrows():  # note 1st row is special, not an actual entry
            id_, pid = row.id, row.parent_id
            added = False
            for parent in graph:
                if parent == pid:
                    graph[parent].append(id_)
                    can_have_child = row.type in ['workset', 'workette']  # types for root-level group, internal group
                    # shouldn't raise an error on `append` if api call well-formed
                    graph[id_] = [] if can_have_child else None
                    added = True
                    break
            if not added:  # should not happen, assuming API returns in correct order
                raise ValueError(f'Parent for node with id={logi(id_)} not found')
        return graph

    def clean_single(self, data_path: str, save: bool = False) -> Tuple[pd.DataFrame, Dict]:
        """
        Clean up raw data for a single date
        """
        path = os_join(self.dataset_path, data_path)
        if self.verbose:
            self.logger.info(f'Cleaning {path}... ')
        df = pd.read_csv(path)
        df = df.apply(DataCleaner._clean_single_entry, axis=1)

        root_id = df.loc[0, 'id']
        graph = DataCleaner._cleaned_df2graph(df)

        def id2txt(i_: str) -> str:
            txts = df.loc[df.id == i_, 'text'].values
            assert len(txts) == 1
            txt = txts[0]
            if isinstance(txt, float) and math.isnan(txt):
                return 'root'
            else:
                return txt
        graph_txt = {id2txt(k): ([id2txt(i) for i in v] if v else v) for k, v in graph.items()}
        path = {n: path2root(graph, root_id, n) for n in graph}  # node => path from root to node
        meta = dict(
            graph=dict(id=graph, name=graph_txt),
            path=dict(id=path, name={id2txt(k): ([id2txt(i) for i in v] if v else v) for k, v in path.items()}),
            # a human-readable snapshot
            tree=dict(
                id=readable_tree(graph, root_id),
                name=readable_tree(graph_txt, id2txt(root_id), parent_prefix='p')
            )
        )
        meta['path-exclusive'] = {
            typ: {k: v[1:-1] if v else None for k, v in meta['path'][typ].items()}
            for typ in meta['path']
        }
        # note since label based on path, the order in label list implies nested level
        id2lbs = {k: [id2txt(i) for i in v] if v else None for k, v in get(meta, 'path-exclusive.id').items()}
        df['labels'] = df.id.apply(lambda i: id2lbs[i])

        if save:
            if os.sep in data_path:
                parent_dirs = data_path.split(os.sep)[:-1]
                os.makedirs(os_join(self.output_path, *parent_dirs), exist_ok=True)
            out_path = os_join(self.output_path, data_path)
            df.to_csv(out_path, index=False)
            out_path = out_path.removesuffix('.csv') + '.json'
            with open(out_path, 'w') as f:
                json.dump(meta, f, indent=4)
        return df, meta

    def clean_all(self, user_id: str, save: bool = False) -> List[Tuple[pd.DataFrame, Dict]]:
        """
        Clean up raw data for a single user
        """
        dates = self.uid2dt[user_id]
        it = tqdm(dates, desc=f'Cleaning up user {logi(user_id)}', unit='date')
        ret = []
        for d in it:
            it.set_postfix(date=logi(d))
            fnm = os_join(user_id, f'{d}.csv')
            ret.append(self.clean_single(fnm, save))
        return ret


if __name__ == '__main__':
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
            uids = get_user_ids(split='prod')[2:]
            # st = '2020-09-01'  # This got error from Myca API
            st = '2020-10-01'
            # st = '2021-01-01'
            # st = '2021-10-28'
            for i in uids:
                dw.get_all(user_id=i, start_date=st, end_date='2022-08-01', save=True)
    # write_all()

    def clean_up():
        dnm = 'raw, 2022-08-10_09-57-34'
        path = os_join(u.dset_path, dnm)
        mic(path)
        mic(os.listdir(path))
        dc = DataCleaner(dataset_path=path, verbose=False)
        user_id = dc.user_ids[0]

        def single():
            date = dc.uid2dt[user_id][-1]
            fnm = os_join(user_id, f'{date}.csv')
            sv = False
            # sv = True
            dc.clean_single(data_path=fnm, save=sv)
        # single()

        def all_():
            # dc.clean_all(user_id=user_id, save=True)
            uids = get_user_ids()
            for i in uids:
                dc.clean_all(i, save=True)
        all_()
    clean_up()
