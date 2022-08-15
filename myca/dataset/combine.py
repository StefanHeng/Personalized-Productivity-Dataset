"""
Aggregate for each user, across dates, into collection of unique action entries

A finalized version for the dataset
"""

import os
import glob
from os.path import join as os_join
from typing import Tuple, List, Dict
from dataclasses import dataclass, asdict
from collections import defaultdict

import pandas as pd
from tqdm.auto import tqdm

from stefutil import *
from myca.util import *
from myca.dataset.util import *


@dataclass(frozen=True)
class ActionEntry:
    """
    Fields for internal processing

    .. note:
    1. `id` will be different for each API call
        Based on observation from seemingly same elements
    2. `parent_id` will no longer be needed with a hierarchical structure in json file, by dates
    3. A `date` field will be added afterwards for the final dataset, based on processing consecutive days
    """
    text: str = None
    note: str = None
    link: str = None
    # creation_time: str = None  # can't be a field cos it messes up hashing, will be added to the final dataset
    type: str = None
    parent_is_group: bool = False
    labels: str = None  # Keep as json string for hashing

    # def __eq__(self, other):
    #     if isinstance(other, ActionEntry):
    #         return self.text == other.text and self.note == other.note and self.link == other.link and \
    #             self.type == other.type and self.parent_is_group == other.parent_is_group and \
    #             self.labels == other.labels  # ignore `creation_time` cos it depends on the queried date
    #     else:
    #         raise TypeError(f'Cannot compare {logi(type(self))} with {logi(type(other))}')


class DataAggregator:
    def __init__(
            self, dataset_path: str, output_path: str = os_join(u.dset_path, f'aggregated, {now(for_path=True)}'),
            verbose: bool = True
    ):
        self.dataset_path = dataset_path
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        self.user_ids = [stem(f) for f in sorted(glob.iglob(os_join(dataset_path, '*')))]
        self.uid2dt: Dict[str, List[str]] = {
            uid: [p for p in sorted(glob.glob(os_join(dataset_path, uid, '*.csv')))]
            for uid in self.user_ids
        }

        self.logger = get_logger(self.__class__.__qualname__)
        self.verbose = verbose

    def aggregate_single(self, user_id: str, save: bool = False) -> pd.DataFrame:
        paths = self.uid2dt[user_id]
        added_entries = set()
        # `creation_time` stored outside of `ActionEntry` so that hashing works
        date2entries: Dict[str, List[ActionEntry]] = defaultdict(list)
        date2creation_time: Dict[str, List[str]] = defaultdict(list)

        it = tqdm(paths, desc=f'Processing {logi(user_id)}', unit='date')
        for p in it:
            date = stem(p)
            df = pd.read_csv(p)
            for e, t in DataAggregator._df2entries(df):
                if e not in added_entries:  # can't use `set` for
                    added_entries.add(e)
                    date2entries[date].append(e)
                    date2creation_time[date].append(t)
            it.set_postfix(dict(n=len(added_entries), added=len(date2entries[date])))
            if date == '2020-10-05':  # TODO: debugging
                break
        # TODO: do I include the date?
        df = pd.DataFrame(sum([[asdict(e) | dict(date=d) for e in lst] for d, lst in date2entries.items()], start=[]))
        # df = pd.DataFrame(sum([[asdict(e) for e in lst] for d, lst in date2entries.items()], start=[]))
        df['creation_time'] = sum(date2creation_time.values(), start=[])
        df = df[['text', 'note', 'link', 'creation_time', 'type', 'parent_is_group', 'labels', 'date']]
        mic(df)

        # sanity check no duplicates
        from collections import Counter
        c = Counter(df.text)
        c = Counter({k: v for k, v in c.items() if v > 1})
        mic(c)
        for txt, cnt in c.most_common():
            rows = df[df['text'] == txt]
            mic(rows)
            r1, r2 = rows.iloc[0], rows.iloc[1]
            r1.creation_time, r2.creation_time = None, None  # Ignore for comparison
            # mic(r1, r3)
            # mic(r1.note, r3.note)
            with pd.option_context('max_colwidth', 90):
                mic(r1.compare(r2))
            mic('')
            # exit(1)

        if save:
            df.to_csv(os_join(self.output_path, f'{user_id}.csv'), index=False)
        return df

    @staticmethod
    def _parse_nan(x):
        return None if is_nan(x) else x

    @staticmethod
    def _df2entries(df: pd.DataFrame) -> List[Tuple[ActionEntry, str]]:
        return [
            (ActionEntry(
                # text=DataAggregator._parse_nan(row['text']),
                # note=DataAggregator._parse_nan(row['note']),
                # link=DataAggregator._parse_nan(row['link']),
                # type=DataAggregator._parse_nan(row['type']),
                # parent_is_group=DataAggregator._parse_nan(row['parent_is_group']),
                # labels=DataAggregator._parse_nan(row['labels'])
                **{k: DataAggregator._parse_nan(row[k]) for k in [
                    'text', 'note', 'link', 'type', 'parent_is_group', 'labels'
                    # 'text', 'note', 'link', 'creation_time', 'type', 'parent_is_group', 'labels'
                ]}
            ), row.creation_time
        ) for _, row in df.iterrows()]


if __name__ == '__main__':
    dnm = 'cleaned, 2022-08-15_10-49-23'
    path = os_join(u.dset_path, dnm)
    da = DataAggregator(dataset_path=path)

    def check_single():
        user_id = da.user_ids[3]
        da.aggregate_single(user_id=user_id)
    check_single()

    def check_data_class():
        e1, e2, e3 = ActionEntry(note='a'), ActionEntry(note='a'), ActionEntry(note='b')
        mic(e1 == e2)
        mic(e1 == e3)
    # check_data_class()
