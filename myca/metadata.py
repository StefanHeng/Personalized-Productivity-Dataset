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


@dataclass(frozen=True)
class StatsOutput:
    stats: Dict[str, Any] = None
    collection: Dict[str, Any] = None


def get_stats(dataset: Dict[str, pd.DataFrame], save: bool = False) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    def _get_single(df_: pd.DataFrame) -> StatsOutput:
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
                level2n_uniq_label=level2n_uniq_label  # #unique labels for each level
            ),
            collection={
                'uniq-labels': uniq_labels,  # list of unique labels across all entries, ordered by frequency
                'level2uniq-labels': level2uniq_labels  # list of unique labels for each level, ordered by frequency
            }
        )
    rows, meta = [], dict()
    for uid, df in dataset.items():
        o = _get_single(df)
        rows.append(dict(user_id=uid) | o.stats)
        meta[uid] = o.collection
    if save:
        date = now(fmt="short-date")
        pd.DataFrame(rows).to_csv(os_join(u.dset_path, f'stats, {date}.csv'), index=False)
        with open(os_join(u.dset_path, f'stats-meta, {date}.json'), 'w') as f:
            json.dump(meta, f, indent=4)
    return pd.DataFrame(rows), meta


if __name__ == '__main__':
    def check_stats():
        dnm = 'aggregated, 2022-08-18_15-27-48'
        path = os_join(u.dset_path, dnm)

        paths = [f for f in sorted(glob.iglob(os_join(path, '*.csv')))]
        d = {stem(p): pd.read_csv(p) for p in paths}
        # mic(d)

        sv = True
        mic(get_stats(d, save=sv))
    check_stats()
