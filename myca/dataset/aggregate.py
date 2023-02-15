"""
Aggregate for each user, across dates, into collection of unique action entries

A finalized version for the dataset
"""

import os
import glob
import json
from os.path import join as os_join
from typing import Tuple, List, Set, Dict, Callable, Union, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

import pandas as pd
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

from stefutil import *
from myca.util import *
from myca.dataset.util import *
from myca.dataset.clean import readable_tree, path2root


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
    workset_only_labels: str = None
    is_focus: bool = None


@dataclass
class Hierarchy:
    graph: AdjList = None  # adjacency list
    idx2name: Dict[int, str] = None  # Mapping from index to category name
    root_index: int = None

    def json_readable(self) -> Dict[str, Any]:
        return {
            'hierarchy_graph': self.graph,
            'vocabulary': self.idx2name,
            'readable': readable_tree(self.graph, root=self.root_index, elm_map=lambda i: self.idx2name[i])
        }


@dataclass
class DatesHierarchyPair:
    dates: List[str] = None
    hierarchy: Hierarchy = None


@dataclass
class AggregateOutput:
    table: pd.DataFrame = None
    hierarchy: Dict[str, Dict[str, Union[AdjList, Tuple]]] = None


def _get_all_tags(rich_txt: str) -> Set[str]:
    soup = BeautifulSoup(rich_txt, 'html.parser')
    s = set()
    for e in soup.find_all():
        s.add(e.name)
    return s


class DataAggregator:
    _meaningless_tags = {'p', 'br'}
    IGNORE_IS_FOCUS = True

    def __init__(
            self, dataset_path: str,
            output_path: str = os_join(u.dset_path, f'{now(fmt="short-date")}_Aggregated-Dataset'),
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
                    'text', 'note', 'link', 'type', 'is_focus', 'parent_is_group', 'labels', 'workset_only_labels'
                ]}
            ), row.creation_time
        ) for _, row in df.iterrows()]

    @staticmethod
    def _get_hierarchy(meta: Dict[str, Any]) -> Hierarchy:
        """
        :param meta: `graph` field of the cleaned version, for a given user on a given date
        :return: Adjacency list for all categorical nodes, i.e. ignoring children

        If the structure without all leaf nodes are the same for 2 consecutive days, consider no change in hierarchy
        """
        # the `name` version of `graph` may be inaccurate, see `DataCleaner.clean_single`
        graph, id2idx, id2nm = meta['index'], meta['index_vocabulary'], meta['id_vocabulary']
        graph = {int(k): v for k, v in graph.items()}  # since json only supports string keys
        idx2id = {v: k for k, v in id2idx.items()}  # Should be 1-to-1 mapping by construction
        # remove leaf nodes
        sub_g = {node: children for node, children in graph.items() if children and len(children) > 0}
        leafs = set(graph.keys()) - set(sub_g.keys())
        # remove edges to leave nodes
        sub_g = {node: [c for c in children if c not in leafs] for node, children in sub_g.items()}

        # sanity check categories still make a complete tree
        root_id = [i for i, nm in id2nm.items() if nm == ROOT_HIERARCHY_NAME]
        assert len(root_id) == 1  # sanity check one root
        root_idx = id2idx[root_id[0]]

        for n in sub_g:
            assert path2root(graph=sub_g, root=root_idx, target=n) is not None
        return Hierarchy(graph=sub_g, idx2name={idx: id2nm[idx2id[idx]] for idx in sub_g.keys()}, root_index=root_idx)

    def _dedup_wrapper(
            self, df: pd.DataFrame, dedup_columns: List[str] = None,
            group_callback: Callable[[pd.DataFrame, Dict[str, Any], List[int]], List[int]] = None,
            strict: bool = True, focus_column: str = None
    ) -> pd.DataFrame:
        """
        Wrapper for all deduplication logic

        :param df: dataframe to dedup
        :param dedup_columns: columns to consider for deduplication
        :param group_callback: callback to each duplicate group
            Takes in
                1. the dataframe
                2. the duplicated keys in the order of `dedup_columns` as dict
                3. the indices of the duplicated rows w.r.t. `df`
            should return indices to drop from `df`, as a subset of the given duplicate indices
        """
        dups_all = df[df.duplicated(subset=dedup_columns, keep=False)]
        if not dups_all.empty:
            n_dup = len(dups_all)
            grps = dups_all.groupby(dedup_columns).groups
            idxs_rmv = []

            for grp_key, grp_idxs in grps.items():
                idxs_rmv += group_callback(df, dict(zip(dedup_columns, grp_key)), grp_idxs.to_list())
            df = df.drop(idxs_rmv)
            df = df.reset_index(drop=True)
            if strict:
                assert not (df.duplicated(subset=dedup_columns, keep=False)).any()  # sanity check
            if focus_column:
                e_str = f'{pl.i(focus_column)} entries'
            else:
                e_str = 'entries'
            self.logger.info(f'{pl.i(n_dup)} duplicate {e_str} reduced to {pl.i(n_dup - len(idxs_rmv))}')
        return df

    def _dedup_label_change(self, df: pd.DataFrame, key: Dict[str, Any], grp_idxs: List[int]) -> List[int]:
        # the last element should be the final label since `creation_time` aligns with df iteration order,
        # just to be safe
        idx_keep = max(grp_idxs, key=lambda i: str_time2time(df.iloc[i]['creation_time']))

        if self.verbose:
            labels = [(None if lb is None else json.loads(lb)) for lb in df.iloc[grp_idxs]['labels']]
            self.logger.info(f'Duplicate key {pl.i(key)} resolved with {pl.s("labels", c="m")} {pl.i(labels)}')
        grp_idxs.remove(idx_keep)
        return grp_idxs

    def _dedup_is_focus_change(self, df: pd.DataFrame, key: Dict[str, Any], grp_idxs: List[int]) -> List[int]:
        # just take the last one, whichever doesn't seem to matter much
        idx_keep = max(grp_idxs, key=lambda i: str_time2time(df.iloc[i]['creation_time']))

        if self.verbose:
            # mic(grp_idxs, df.iloc[grp_idxs]['is_focus'])
            # exit(1)
            is_focus = [(None if lb is None else json.loads(lb)) for lb in df.iloc[grp_idxs]['labels']]
            self.logger.info(f'Duplicate key {pl.i(key)} resolved with {pl.s("is_focus", c="m")} {pl.i(is_focus)}')
        grp_idxs.remove(idx_keep)
        return grp_idxs

    @staticmethod
    def no_link(val) -> bool:
        return val is None or val == '[]'

    @staticmethod
    def _sanity_check_remove_none(
            df: pd.DataFrame, grp_idxs: List[int], flags: pd.Series, strict: bool = True
    ) -> List[int]:
        """
        During duplicate removal, if the field in question for **some** row is meaningful,
            remove the fields that are meaningless

        :return: Indices to remove
        """
        rows = df.iloc[grp_idxs]
        if strict:
            idxs_none = flags.index[flags].to_list()
            times = rows.creation_time.map(str_time2time)[~flags]
            # the rows with meaningless field should appear at the earliest date than the rows with field
            # TODO: not always the case??
            for i_none in idxs_none:
                assert str_time2time(df.iloc[i_none].creation_time) < times.min()
        return rows[flags].index.to_list()  # drop the row with None

    def _dedup_link_semi_same(self, df: pd.DataFrame, key: Dict[str, Any], grp_idxs: List[int]) -> List[int]:
        links = df.iloc[grp_idxs]['link']
        flags = links.map(lambda lk: DataAggregator.no_link(lk))
        if flags.all():
            idx_keep = max(grp_idxs, key=lambda i: str_time2time(df.iloc[i]['creation_time']))
            grp_idxs.remove(idx_keep)

            if self.verbose:
                links = links.values.tolist()
                self.logger.info(f'Duplicate key {pl.i(key)} resolved with {pl.s("links", c="m")} {pl.i(links)}')
            return grp_idxs
        else:  # some or all `links` field is meaningful
            if flags.any():
                return DataAggregator._sanity_check_remove_none(df, grp_idxs, flags, strict=False)
            else:
                return []  # no rows should be dropped

    def _dedup_type_change(self, df: pd.DataFrame, key: Dict[str, Any], grp_idxs: List[int]) -> List[int]:
        types = df.iloc[grp_idxs]['type']
        if types.map(lambda t: t is None or t in ['workset', 'workette']).all():
            idx_keep = max(grp_idxs, key=lambda i: str_time2time(df.iloc[i]['creation_time']))
            grp_idxs.remove(idx_keep)

            if self.verbose:
                types = types.values.tolist()
                self.logger.info(f'Duplicate key {pl.i(key)} resolved with {pl.s("types", c="m")} {pl.i(types)}')
            return grp_idxs
        else:  # some or all `types` field is special
            flags = types.map(lambda t: t is None)
            if flags.any():  # e.g. `None` => `note`
                return DataAggregator._sanity_check_remove_none(df, grp_idxs, flags)
            else:
                return []

    @staticmethod
    def no_note(val) -> bool:
        return val is None or _get_all_tags(val) == DataAggregator._meaningless_tags

    def _dedup_note_semi_same(self, df: pd.DataFrame, key: Dict[str, Any], grp_idxs: List[int]) -> List[int]:
        rows = df.iloc[grp_idxs]
        notes = rows['note']
        flags = notes.map(lambda nt: DataAggregator.no_note(nt))

        if flags.all():
            idx_keep = max(grp_idxs, key=lambda i: str_time2time(df.iloc[i]['creation_time']))
            grp_idxs.remove(idx_keep)

            if self.verbose:
                notes = notes.values.tolist()
                self.logger.info(f'Duplicate key {pl.i(key)} resolved with {pl.s("notes", c="m")} {pl.i(notes)}')
            return grp_idxs
        else:  # some `notes` field is meaningful
            # TODO: deal with multiple meaningful notes
            return rows[flags].index.to_list()  # just drop the rows with meaningless notes

    def aggregate_single(self, user_id: str, save: bool = False) -> Tuple[pd.DataFrame, Dict]:
        paths = self.uid2dt[user_id]
        added_entries = set()
        # `creation_time` stored outside `ActionEntry` so that hashing works
        date2entries: Dict[str, List[ActionEntry]] = defaultdict(list)
        date2creation_time: Dict[str, List[str]] = defaultdict(list)

        it = tqdm(paths, desc=f'Merging entries for user {pl.i(user_id)}', unit='date')
        for p in it:
            date = stem(p)
            df = pd.read_csv(p)
            for e, t in DataAggregator._df2entries(df.iloc[1:]):  # remove 1st row since not an actual entry
                if e not in added_entries:  # can't use `set` for
                    added_entries.add(e)
                    date2entries[date].append(e)
                    date2creation_time[date].append(t)
            it.set_postfix(dict(n=pl.i(len(added_entries)), added=pl.i(len(date2entries[date]))))
            # if date == '2020-10-20':  # TODO: debugging
            #     break
        # include the date since the `creation_time` field is influenced by time zone,
        # e.g. timestamp of 10-06 may appear when date 10-05 is queried
        df = pd.DataFrame(sum([[asdict(e) | dict(date=d) for e in lst] for d, lst in date2entries.items()], start=[]))
        df['creation_time'] = sum(date2creation_time.values(), start=[])
        # cols = ['text', 'note', 'link', 'creation_time', 'type', 'parent_is_group', 'labels', 'workset_only_labels']
        # df = df[cols + ['date']]
        cols = ['text', 'note', 'link', 'creation_time', 'type', 'is_focus', 'parent_is_group', 'labels']
        df = df[cols + ['workset_only_labels', 'date']]

        # only the `labels` changed, i.e. the exact same entry is moved in the hierarchy
        # => keep the last modified version
        # note that we ignore `date` field for dedup
        # set of properties that uniquely identify an entry:
        #   ['text', 'note', 'link', 'type', 'is_focus', 'parent_is_group', 'labels']
        # for now, just ignore `is_focus` field for dedup, TODO
        # otherwise, in case e.g. `type` and `is_focus` changes at the same time, both entries are kept
        ignore_focus = DataAggregator.IGNORE_IS_FOCUS
        cols = ['text', 'note', 'link', 'type'] if ignore_focus else ['text', 'note', 'link', 'type', 'is_focus']
        df = self._dedup_wrapper(df, dedup_columns=cols, group_callback=self._dedup_label_change, focus_column='labels')
        # only difference is `link` change, if the change is between `None` and `[]` consider as duplicates
        # => keep the last modified version
        if ignore_focus:
            cols = ['text', 'note', 'type', 'parent_is_group', 'labels']
        else:
            cols = ['text', 'note', 'type', 'is_focus', 'parent_is_group', 'labels']
        df = self._dedup_wrapper(df, dedup_columns=cols, group_callback=self._dedup_link_semi_same, focus_column='link')
        # only difference is `type`, pbb cos user cleans up the hierarchy => keep the last modified version
        if ignore_focus:
            cols = ['text', 'note', 'link', 'parent_is_group', 'labels']
        else:
            cols = ['text', 'note', 'link', 'is_focus', 'parent_is_group', 'labels']
        df = self._dedup_wrapper(
            df, dedup_columns=cols, group_callback=self._dedup_type_change, focus_column='type', strict=False
        )
        # `note` changes, from None to a rich text with no real content, consider as duplicates
        if ignore_focus:
            cols = ['text', 'type', 'link', 'parent_is_group', 'labels']
        else:
            cols = ['text', 'link', 'type', 'is_focus', 'parent_is_group', 'labels']
        df = self._dedup_wrapper(
            df, dedup_columns=cols, group_callback=self._dedup_note_semi_same, focus_column='note', strict=False
        )
        if not ignore_focus:
            df = self._dedup_wrapper(
                df, dedup_columns=['text', 'note', 'link', 'type', 'parent_is_group', 'labels'],
                group_callback=self._dedup_is_focus_change, focus_column='is_focus'
            )

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

        def date2meta(date_: str):
            with open(os_join(self.dataset_path, user_id, f'{date_}.json'), 'r') as fl:
                return json.load(fl)['graph']
        date2hierarchy = {dt: DataAggregator._get_hierarchy(meta=date2meta(dt)) for dt in sorted(date2entries.keys())}
        dates = iter(sorted(date2entries.keys()))
        d = next(dates, None)
        assert d is not None
        dates2hier: List[DatesHierarchyPair] = [DatesHierarchyPair(dates=[d], hierarchy=date2hierarchy[d])]
        curr_dates = [d]
        d = next(dates, None)
        while d is not None:
            last_pair = dates2hier[-1]
            hier_curr = date2hierarchy[d]
            if hier_curr == last_pair.hierarchy:  # same hierarchy compared to last added date
                curr_dates.append(d)
                dates2hier[-1].dates = curr_dates
            else:
                dates2hier.append(DatesHierarchyPair(dates=[d], hierarchy=hier_curr))
                curr_dates = [d]
            d = next(dates, None)
        dates2hier: Dict[Tuple, Hierarchy] = {tuple(pr.dates): pr.hierarchy for pr in dates2hier}
        # root node name, see `clean.Id2Text`
        dates2meta = {k: h.json_readable() for k, h in dates2hier.items()}
        d_log = {'#entry': len(df), '#hiearchy': len(dates2hier)}
        self.logger.info(f'Aggregation for user {pl.i(user_id)} completed with {pl.i(d_log)}')

        if save:
            df.to_csv(os_join(self.output_path, f'{user_id}.csv'), index=False)
            with open(os_join(self.output_path, f'{user_id}.json'), 'w') as f:  # tuple not json serializable
                json.dump({json.dumps(k): v for k, v in dates2meta.items()}, f, indent=4)
        return df, dates2meta

    def aggregate(self, save: bool = False) -> Dict[str, Tuple[pd.DataFrame, Dict]]:
        return {uid: self.aggregate_single(uid, save) for uid in self.user_ids}


if __name__ == '__main__':
    from myca.dataset.clean import ROOT_HIERARCHY_NAME

    dnm = '23-02-10_Cleaned-Dataset'
    path = os_join(u.dataset_path, dnm)
    da = DataAggregator(dataset_path=path, root_name=ROOT_HIERARCHY_NAME)

    def check_single():
        da.verbose = True
        # user_id = da.user_ids[3]  # most entries/day
        user_id = da.user_ids[2]  # least entries/day
        df = da.aggregate_single(user_id=user_id)[0]
        mic(len(df))
    # check_single()

    def run():
        d = da.aggregate(save=True)
        mic({k: len(v[0]) for k, v in d.items()})
    run()

    def check_data_class():
        e1, e2, e3 = ActionEntry(note='a'), ActionEntry(note='a'), ActionEntry(note='b')
        mic(e1 == e2)
        mic(e1 == e3)
    # check_data_class()
