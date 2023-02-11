"""
Get statistics for each user
"""

import re
import json
import glob
from os.path import join as os_join
from typing import List, Tuple, Dict, Any, Union, Iterable
from collections import defaultdict, Counter
from dataclasses import dataclass

import pandas as pd

from stefutil import *
from myca.util import *
from myca.dataset import is_nan


def to_tokens(x: Union[str, float]) -> List[str]:
    if not hasattr(to_tokens, 'token_pattern'):
        to_tokens.token_pattern = re.compile(r'(?u)\b\w+\b')  # taken from sklearn.CountVectorizer
    if isinstance(x, str):
        return list(to_tokens.token_pattern.findall(x))
    else:
        assert is_nan(x)
        return []


def _save(df: pd.DataFrame = None, meta: Dict = None, name: str = None):
    date = now(fmt="short-date")
    df.to_csv(os_join(u.dataset_path, f'{date}_{name}.csv'), index=False)
    with open(os_join(u.dataset_path, f'{date}_{name}, meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)
    return df


def get_raw_stats(dataset: Dict[str, Dict[str, pd.DataFrame]], save: bool = True):
    """
    Stats based on `cleaned` dataset, date-level stats then average
    """
    def _get_single_user(date2df):
        _rows = []
        date2uniq_labels = dict()
        last_uniq_labels = set()

        for dt, df_ in date2df.items():
            all_labels: List[Tuple[str]] = [tuple(lbs) for lbs in df_.labels[df_.labels.notnull()].map(json.loads)]
            uniq_labels = list(Counter(all_labels).keys())  # ordered by frequency

            _rows.append(dict(
                n_entries=len(df_),
                depth=max(len(lbs) for lbs in all_labels),  # Largest nested level of category
                n_uniq_labels=len(uniq_labels)
            ))
            curr_set = set(uniq_labels)
            if curr_set != last_uniq_labels:
                # So that `json` output will be on the same row, easier to inspect
                date2uniq_labels[dt] = pl.nc(uniq_labels)
                last_uniq_labels = curr_set
        return pd.DataFrame(_rows).mean(axis=0).to_dict(), date2uniq_labels  # column-wise average
    rows, meta = [], dict()
    for uid, d2d in dataset.items():
        row, m = _get_single_user(d2d)
        row['#days of hierarchy change'] = len(m)
        rows.append(dict(user_id=uid) | row)
        meta[uid] = {'date2uniq-labels': m}
    df = pd.DataFrame(rows)
    if save:
        _save(df=df, meta=meta, name='Raw Stats')
    return df, meta


@dataclass(frozen=True)
class StatsOutput:
    stats: Dict[str, Any] = None
    collection: Dict[str, Any] = None


def get_aggregated_stats(
        dataset: Dict[str, pd.DataFrame], save: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Stats based on the already aggregated dataset, a set of labels for a given user across all dates
    """
    def _get_single_user_row(df_: pd.DataFrame) -> StatsOutput:
        n_entries = len(df_)
        n_text_token = sum(len(to_tokens(x)) for x in df_.text)
        n_text_uniq_token = len(set().union(*[to_tokens(x) for x in df_.text]))
        avg_n_text_token = n_text_token / n_entries
        n_day = len(set(df_.date))

        all_labels: Iterable[List[str]] = df_.labels[df_.labels.notnull()].map(json.loads)
        all_label: List[str] = sum(all_labels, start=[])
        n_labels_token = sum(len(to_tokens(x)) for x in all_label)
        avg_n_labels_token = n_labels_token / n_entries
        n_label = len(all_label)
        avg_n_label = n_label / n_entries
        label2cnt = Counter(all_label)
        uniq_labels = list(label2cnt.keys())  # Order by frequency
        n_uniq_label = len(label2cnt)
        level2label2cnt = defaultdict(Counter)
        for lbs in all_labels:
            for lvl, lb in enumerate(lbs):
                level2label2cnt[lvl][lb] += 1
        level2uniq_labels = {lvl: list(l2c.keys()) for lvl, l2c in level2label2cnt.items()}
        level2n_uniq_label = {lvl: len(l2c) for lvl, l2c in level2label2cnt.items()}
        return StatsOutput(
            stats=dict(
                n_day=n_day,  # #days the myca snapshot changes
                n_entries=n_entries,  # #action entries
                n_text_token=n_text_token,  # total #token for all entries
                n_text_uniq_token=n_text_uniq_token,  # #unique token across all entries
                avg_n_text_token=avg_n_text_token,  # avg #token for each entry
                n_label=n_label,  # #labels for all entries
                avg_n_label=avg_n_label,  # avg #label for each entry
                avg_n_labels_token=avg_n_labels_token,  # avg #token for each label
                n_uniq_label=n_uniq_label,  # #unique labels across all entries
                level2n_uniq_label=level2n_uniq_label,  # #unique labels for each level
                labels=pl.nc(uniq_labels)
            ),
            collection={
                'uniq-labels': pl.nc(uniq_labels),  # list of unique labels across all entries, ordered by frequency
                # list of unique labels for each level, ordered by frequency
                'level2uniq-labels': {k: pl.nc(v) for k, v in level2uniq_labels.items()}
            }
        )
    rows, meta = [], dict()
    for uid, df in dataset.items():
        o = _get_single_user_row(df)
        rows.append(dict(user_id=uid) | o.stats)
        meta[uid] = o.collection
    df = pd.DataFrame(rows)
    if save:
        _save(df=df, meta=meta, name='Aggregated Stats')
    return df, meta


if __name__ == '__main__':
    from tqdm.auto import tqdm

    def load_cleand_dset(dataset_name: str = '22-09-02_Cleaned-Dataset'):
        path = os_join(u.dataset_path, dataset_name)
        user_ids = [stem(f) for f in sorted(glob.iglob(os_join(path, '*')))]
        uid2paths: Dict[str, List[str]] = {
            uid: [p for p in sorted(glob.glob(os_join(path, uid, '*.csv')))]
            for uid in user_ids
        }
        dset = dict()
        for uid, v in uid2paths.items():
            it = tqdm(v, desc=f'Loading DataFrames for User {pl.i(uid[:4])}')
            d: Dict[str, pd.DataFrame] = dict()
            for p in it:
                date = stem(p)
                it.set_postfix(date=pl.i(date))
                d[date] = pd.read_csv(p)
            dset[uid] = d
        return dset

    def check_raw():
        dset = load_cleand_dset()
        get_raw_stats(dataset=dset)
    # check_raw()

    def check_aggregated():
        dnm = '22-09-02_Aggregated-Dataset'
        path = os_join(u.dataset_path, dnm)

        paths = [f for f in sorted(glob.iglob(os_join(path, '*.csv')))]
        d = {stem(p): pd.read_csv(p) for p in paths}
        # mic(d)

        sv = True
        mic(get_aggregated_stats(d, save=sv))
    check_aggregated()
