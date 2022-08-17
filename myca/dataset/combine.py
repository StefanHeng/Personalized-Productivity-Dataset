"""
Aggregate for each user, across dates, into collection of unique action entries

A finalized version for the dataset
"""

import os
import glob
import json
from os.path import join as os_join
from typing import Tuple, List, Dict, Union
from dataclasses import dataclass, asdict
from collections import defaultdict

import pandas as pd
from tqdm.auto import tqdm

from stefutil import *
from myca.util import *
from myca.dataset.util import *
from myca.dataset.clean import readable_tree


__all__ = ['AggregateOutput', 'DataAggregator']


@dataclass(frozen=True)
class ActionEntry:
    """
    Fields for internal processing

    .. note:
    1. `id` will be different for each API call
        Based on observation from seemingly same elements
    2. `parent_id` will no longer be needed with a hierarchical structure in json file, by dates
    3. A `date` field will be added afterwards for the final dataset, based on processing consecutive days TODO
    """
    text: str = None
    note: str = None
    link: str = None
    # `creation_time` can't be a field cos it messes up hashing, will be added to the final dataset
    type: str = None
    parent_is_group: bool = False
    labels: str = None  # Keep as json string for hashing


@dataclass
class DatesHierarchyPair:
    dates: List[str] = None
    hierarchy: AdjList = None  # adjacency list


@dataclass
class AggregateOutput:
    table: pd.DataFrame = None
    hierarchy: Dict[str, Dict[str, Union[AdjList, Tuple]]] = None


class DataAggregator:
    def __init__(
            self, dataset_path: str, output_path: str = os_join(u.dset_path, f'aggregated, {now(for_path=True)}'),
            verbose: bool = False, root_name: str = 'root'
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

        self.root_name = root_name

    @staticmethod
    def _parse_nan(x):
        return None if is_nan(x) else x

    @staticmethod
    def _df2entries(df: pd.DataFrame) -> List[Tuple[ActionEntry, str]]:
        return [(
            ActionEntry(
                **{k: DataAggregator._parse_nan(row[k]) for k in [
                    'text', 'note', 'link', 'type', 'parent_is_group', 'labels'
                    # 'text', 'note', 'link', 'creation_time', 'type', 'parent_is_group', 'labels'
                ]}
            ), row.creation_time
        ) for _, row in df.iterrows()]

    def _get_hierarchy(self, user_id: str, date: str) -> AdjList:
        """
        :return: adjacency list, i.e. parent -> children

        If the structure without all leaf nodes are the same for 2 consecutive days, consider no change in hierarchy
        """
        with open(os_join(self.dataset_path, user_id, f'{date}.json'), 'r') as f:
            meta = json.load(f)
        graph = get(meta, 'graph.name')
        # remove leaf nodes
        ret = {node: children for node, children in graph.items() if children and len(children) > 0}
        leafs = set(graph.keys()) - set(ret.keys())
        # remove edges to leave nodes
        ret = {node: [c for c in children if c not in leafs] for node, children in ret.items()}
        return ret

    def aggregate_single(self, user_id: str, save: bool = False) -> Tuple[pd.DataFrame, Dict]:
        paths = self.uid2dt[user_id]
        added_entries = set()
        # `creation_time` stored outside of `ActionEntry` so that hashing works
        date2entries: Dict[str, List[ActionEntry]] = defaultdict(list)
        date2creation_time: Dict[str, List[str]] = defaultdict(list)

        it = tqdm(paths, desc=f'Processing user {logi(user_id)}', unit='date')
        for p in it:
            date = stem(p)
            df = pd.read_csv(p)
            for e, t in DataAggregator._df2entries(df.iloc[1:]):  # remove 1st row since not an actual entry
                if e not in added_entries:  # can't use `set` for
                    added_entries.add(e)
                    date2entries[date].append(e)
                    date2creation_time[date].append(t)
            it.set_postfix(dict(n=logi(len(added_entries)), added=logi(len(date2entries[date]))))
            # if date == '2020-10-20':  # TODO: debugging
            #     break
        # include the date since the `creation_time` field is influenced by time zone,
        # e.g. timestamp of 10-06 may appear when date 10-05 is queried
        df = pd.DataFrame(sum([[asdict(e) | dict(date=d) for e in lst] for d, lst in date2entries.items()], start=[]))
        df['creation_time'] = sum(date2creation_time.values(), start=[])
        df = df[['text', 'note', 'link', 'creation_time', 'type', 'parent_is_group', 'labels', 'date']]

        # In such case, the `labels` changed, the exact same entry is moved in the hierarchy
        # we keep the last modified version
        dedup_cols = ['text', 'note', 'link', 'type']
        dups_all = df[df.duplicated(subset=dedup_cols, keep=False)]
        if not dups_all.empty:
            n_dup = len(dups_all)
            idxs_rmv = set(dups_all.index.to_list())
            grps = dups_all.groupby(dedup_cols).groups

            for grp_key, grp_idxs in grps.items():
                grp_idxs = grp_idxs.to_list()
                idx_keep = max(grp_idxs, key=lambda i: str_time2time(df.iloc[i]['creation_time']))
                idxs_rmv.remove(idx_keep)
                # since `creation_time` aligns with df iteration order, the last element should be the final label
                labels = [('' if lb is None else json.loads(lb)) for lb in df.iloc[grp_idxs]['labels']]
                key = dict(zip(dedup_cols, grp_key))
                if self.verbose:
                    self.logger.info(f'Duplicate key {logi(key)} resolved with labels {logi(labels)}')
            df = df.drop(idxs_rmv)
            df = df.reset_index(drop=True)
            assert not (df.duplicated(subset=dedup_cols, keep=False)).any()  # sanity check
            self.logger.info(f'{logi(n_dup)} duplicate entries reduced to {logi(len(grps))}')

        sanity_check = False  # potential "duplicates" all make sense
        # sanity_check = True
        if sanity_check:
            from collections import Counter
            c = Counter(df.text)
            c = Counter({k: v for k, v in c.items() if v > 1 and k is not None})
            mic(c, sum(c.values()))
            for txt, cnt in c.most_common():
                mic(txt, cnt)
                rows = df[df['text'] == txt]
                mic(rows)
                r1, r2 = rows.iloc[0], rows.iloc[1]
                r1.creation_time, r2.creation_time = None, None  # Ignore for comparison
                with pd.option_context('max_colwidth', 90):
                    mic(r1.compare(r2))
                mic('')
            # exit(1)

        # compress hierarchy into necessary changes
        # since date in `Y-m-d`, temporal order maintained
        date2hierarchy = {date: self._get_hierarchy(user_id, date) for date in sorted(date2entries.keys())}
        dates = iter(sorted(date2entries.keys()))
        d = next(dates, None)
        assert d is not None
        dates2hierarchy: List[DatesHierarchyPair] = [DatesHierarchyPair(dates=[d], hierarchy=date2hierarchy[d])]
        curr_dates = [d]
        d = next(dates, None)
        while d is not None:
            last_pair = dates2hierarchy[-1]
            hier_curr = date2hierarchy[d]
            if hier_curr == last_pair.hierarchy:  # same hierarchy compared to last added date
                curr_dates.append(d)
                dates2hierarchy[-1].dates = curr_dates
            else:
                dates2hierarchy.append(DatesHierarchyPair(dates=[d], hierarchy=hier_curr))
                curr_dates = [d]
            d = next(dates, None)
        dates2hierarchy: Dict = {tuple(p.dates): p.hierarchy for p in dates2hierarchy}
        # root node name, see `clean.Id2Text`
        dates2meta = {
            k: dict(hierarchy=h, tree=readable_tree(h, root=self.root_name)) for k, h in dates2hierarchy.items()
        }

        if save:
            df.to_csv(os_join(self.output_path, f'{user_id}.csv'), index=False)
            with open(os_join(self.output_path, f'{user_id}.json'), 'w') as f:  # tuple not json serializable
                json.dump({json.dumps(k): v for k, v in dates2meta.items()}, f, indent=4)
        return df, dates2meta

    def aggregate(self, save: bool = False) -> Dict[str, pd.DataFrame]:
        return {uid: self.aggregate_single(uid, save) for uid in self.user_ids}


if __name__ == '__main__':
    dnm = 'cleaned, 2022-08-17_16-06-12'
    path = os_join(u.dset_path, dnm)
    da = DataAggregator(dataset_path=path, root_name='__ROOT__')

    def check_single():
        # user_id = da.user_ids[3]  # most entries/day
        user_id = da.user_ids[2]  # least entries/day
        da.aggregate_single(user_id=user_id)
    # check_single()

    da.aggregate(save=True)

    def check_data_class():
        e1, e2, e3 = ActionEntry(note='a'), ActionEntry(note='a'), ActionEntry(note='b')
        mic(e1 == e2)
        mic(e1 == e3)
    # check_data_class()
