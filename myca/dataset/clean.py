"""
Cleanup dataset down to the relevant fields w/ human-readable snapshot of data for each date

TODO: sanity check creation time order
"""


import os
import json
import glob
from os.path import join as os_join
from typing import List, Tuple, Dict, Any, Union, Optional, Callable
from dataclasses import dataclass

import pandas as pd
from tqdm.auto import tqdm

from stefutil import *
from myca.util import *
from myca.dataset.util import *


__all__ = ['ROOT_HIERARCHY_NAME', 'Id2Text', 'path2root', 'readable_tree', 'DataCleaner']


ROOT_HIERARCHY_NAME = '__ROOT__'


@dataclass
class CleanOutput:
    table: pd.DataFrame = None
    meta: Dict[str, Dict[str, Any]] = None


class Id2Text:
    def __init__(self, df: pd.DataFrame, enforce_single: bool = True, root_name: str = 'root'):
        self.df = df
        self.enforce_single = enforce_single
        self.root_name = root_name

        if (df.text == self.root_name).any():
            # Resolve, otherwise readable graph representation will be incorrect, cos root node info will be overridden
            raise ValueError(f'Root node name {pl.i(self.root_name)} found in {pl.i("text")} column')
        self.root_id = df.loc[0, 'id']

    def __call__(self, id_: str) -> str:
        if id_ == self.root_id:
            return self.root_name
        else:  # so that root name don't accidentally get returned if `text` is empty
            txts = self.df.loc[self.df.id == id_, 'text'].values
            if self.enforce_single:
                assert len(txts) == 1
            txt = txts[0]
            return '' if is_nan(txt) else txt

    @property
    def vocab(self) -> Dict[str, str]:
        return {id_: self.__call__(id_) for id_ in self.df.id}


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
    if root == target:
        return [root]
    else:
        return _path2root(graph, root, target, [root])  # inclusive start & end


def readable_tree(
        graph: AdjList, root: T = 'root', parent_prefix: str = None,
        elm_map: Callable[[T], Any] = None
) -> Union[Any, Tuple[Any, List[Any]]]:
    """
    :param graph: Adjacency list of a graph/tree
    :param root: Root node
    :param parent_prefix: Needed for precision to differentiate inner & leaf nodes
        Counter example: 2 nodes, A, B are neighbors of each other, A is leaf node, B is inner node
        But they would appear as if A has a single child B
        This is because Tuple is rendered as List in json
    :param elm_map: Map each element in the graph to something else, e.g. index to text
    :return: nested binary tuples of (name, children)
    """
    children = graph[root]
    is_leaf = (not children) or len(children) == 0
    if is_leaf:
        if elm_map:
            root = elm_map(root)
        return root
    else:
        p = f'{parent_prefix}_{root}' if parent_prefix else root
        if elm_map:
            p = elm_map(p)
        return p, [readable_tree(graph, root=c, parent_prefix=parent_prefix, elm_map=elm_map) for c in children]


class Element2Index:
    """
    Assign indices to elements, index increases as items are added to vocabulary
    """
    def __init__(self):
        self.vocab: Dict[str, int] = dict()
        self.idx = 0

    def __call__(self, element: str) -> int:
        if element not in self.vocab:
            self.vocab[element] = self.idx
            self.idx += 1
        return self.vocab[element]


class DataCleaner:
    """
    Clean up the raw dataset (see `DataWriter`) from a given date into our format
    """
    # `item` is `workette`, `group` is `workset`, `item` are front end names,
    # TODO: should not even appear in backend API call in the first place...
    _type_fe2type_be = dict(group='workset', item='workette')

    def __init__(
            self, dataset_path: str,
            output_path: str = os_join(u.dset_path, f'{now(fmt="short-date")}_Cleaned-Dataset'),
            verbose: bool = True, root_name: str = 'root'
    ):
        self.logger = get_logger(self.__class__.__qualname__)
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

        self.root_name = root_name

    def _map_type(self, t: str) -> str:
        if t in DataCleaner._type_fe2type_be:
            t_ = DataCleaner._type_fe2type_be[t]
            self.logger.warning(f'Front End name {pl.i(t)} appeared, converted to {pl.i(t_)}')
            return t_
        else:
            return t

    def _clean_single_entry(self, entry: pd.Series) -> pd.Series:
        return pd.Series(dict(
            id=entry['jid'],  # id of the action entry
            text=entry['context.name'],  # actual text for the action entry
            note=entry['context.note'],
            link=entry.get('context.links', None),  # earlier entries don't contain field `links`
            creation_time=entry['j_timestamp'],
            type=self._map_type(entry['context.wtype']),  # UI type of the entry
            parent_id=entry['field1'],  # by API call design
            is_focus=entry['context.is_MIT']
        ))

    @staticmethod
    def _cleaned_df2graph(df: pd.DataFrame) -> AdjList:
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
                    # workset & workette are types for root-level group, internal group respectively;
                    # Practically anything can have children
                    # TODO: empirically found entries **w/ no type** and w/ children?
                    typ = row.type
                    can_have_child = typ in ['workset', 'workette', 'note', 'link', 'item'] or is_nan(typ)
                    # shouldn't raise an error on `append` if api call well-formed
                    graph[id_] = [] if can_have_child else None
                    added = True
                    break
            if not added:  # should not happen, assuming API returns in correct order
                raise ValueError(f'Parent for node with {pl.i(row.to_dict())} not found')
        return graph

    def clean_df(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, AdjList]:
        df = df.apply(self._clean_single_entry, axis=1)

        # sanity check time in increasing order
        sort_col = 'creation_time_obj'
        times = df[sort_col] = df.creation_time.map(str_time2time)
        # the root node creation_time should be the smallest, i.e. after sorting, should remain in the 1st place
        assert (times[0] < times[1:]).all()
        graph = DataCleaner._cleaned_df2graph(df)  # Need the original order so that parent for all nodes can be found
        # stable sort needed, to respect original API return order in resolving parent,
        # which also empirically make sense, e.g. `__ROOT__` => `Chores`
        df = df.sort_values(by=sort_col, kind='stable').drop(sort_col, axis=1).reset_index(drop=True)

        i2t = Id2Text(df, enforce_single=False, root_name=self.root_name)
        dup_flag = df.id.value_counts() != 1
        if dup_flag.any():
            dup = dup_flag[dup_flag].index.to_list()
            for d in dup:
                # The same id appears in multiple rows, assume it updates the same action entry
                # the update must be moving the action entry, i.e. changing parent
                idxs = sorted(df.index[df.id == d].to_list())
                parent_nms = [i2t(df.loc[idx, 'parent_id']) for idx in idxs]
                d_log = dict(id=d, text=i2t(d), indices=idxs, parent_names=parent_nms)
                self.logger.info(f'Duplicate id found with {pl.i(d_log)}, assumes entry is moved among the hierarchy')
                row0 = df.loc[idxs[0]].drop(labels='parent_id')
                assert all(df.loc[i].drop(labels='parent_id').equals(row0) for i in idxs[1:])
                df = df.drop(idxs[:-1])  # Only keep the bottom-most row, which is the last modified one
            df = df.reset_index(drop=True)
            assert (df.id.value_counts() == 1).all()  # sanity check

        def add_flag(e: pd.Series) -> Optional[bool]:
            """
            :return: Whether the current entry is a **direct** children of a `workset`
                Intended for an easier classification task, as
                    items under a `workette` are often sequential steps of an item
                    items under a `workset` are often collections of independent items
            """
            if is_nan(e.parent_id):  # special case, root node, not an actual item
                return
            else:
                parents = df.loc[df.id == e.parent_id]
                assert len(parents) == 1  # sanity check, should have no duplicate
                return parents.iloc[0].type == 'workset'
        df['parent_is_group'] = df.apply(add_flag, axis=1)
        return df, graph

    def clean_single(self, data_path: str, save: bool = False) -> CleanOutput:
        """
        Clean up raw dataset for a single date
        """
        path = os_join(self.dataset_path, data_path)
        if self.verbose:
            self.logger.info(f'Cleaning {pl.i(path)}... ')
        df = pd.read_csv(path)
        if len(df) == 1:
            self.logger.info(f'No action entries found with {pl.i(data_path)}')
        else:
            df, graph = self.clean_df(df)

            i2t = Id2Text(df, root_name=self.root_name)  # Not necessarily one-to-one mapping
            root_id = i2t.root_id
            e2i = Element2Index()  # Definitely one-to-one mapping

            graph_txt = {i2t(k): ([i2t(i) for i in v] if v is not None else v) for k, v in graph.items()}
            graph_idx = {e2i(k): ([e2i(i) for i in v] if v is not None else v) for k, v in graph.items()}

            # sanity check forms a valid tree
            for n in graph.keys():
                assert path2root(graph=graph, root=root_id, target=n) is not None
            rood_idx = e2i(root_id)
            for n in graph_idx.keys():
                assert path2root(graph=graph_idx, root=rood_idx, target=n) is not None

            path = {n: path2root(graph, root_id, n) for n in graph}  # node => path from root to node
            meta = dict(
                # !! Note that `name` version of `graph` may not be accurate if multiple categories share the same name
                # For an intermediate representation on readability & precision, use the `index` version
                graph=dict(
                    id=graph, id_vocabulary=i2t.vocab, name=graph_txt,
                    index=graph_idx, index_vocabulary=e2i.vocab
                ),
                path=dict(id=path, name={i2t(k): ([i2t(i) for i in v] if v else v) for k, v in path.items()}),
                # a human-readable snapshot
                tree=dict(
                    id=readable_tree(graph, root_id),
                    name=readable_tree(graph_txt, i2t(root_id), parent_prefix='p')
                )
            )
            meta['path-exclusive'] = {
                typ: {k: v[1:-1] if v else None for k, v in meta['path'][typ].items()}
                for typ in meta['path']
            }
            # note since label based on path, the order in label list implies nested level
            id2lbs = {k: [i2t(i) for i in v] if v else None for k, v in get(meta, 'path-exclusive.id').items()}
            df['labels'] = df.id.apply(lambda i: id2lbs[i])

            lb2is_group = {row.text: row.type == 'workset' for i, row in df.iloc[1:].iterrows()}  # skip 1st special row

            def label_map(row: pd.Series) -> Optional[List[str]]:
                labels = row.labels
                if labels is None:
                    return
                else:
                    ret = []
                    for lb in labels:
                        if lb2is_group[lb]:  # must exist in the dict by construction
                            ret.append(lb)
                        else:
                            break
                    return ret
            df['workset_only_labels'] = df.apply(label_map, axis=1).map(json.dumps)  # up until all `group` names
            df.labels = df.labels.map(json.dumps)

            if save:
                if os.sep in data_path:
                    parent_dirs = data_path.split(os.sep)[:-1]
                    os.makedirs(os_join(self.output_path, *parent_dirs), exist_ok=True)
                out_path = os_join(self.output_path, data_path)
                df.to_csv(out_path, index=False)
                out_path = out_path.removesuffix('.csv') + '.json'
                with open(out_path, 'w') as f:
                    json.dump(meta, f, indent=4)
            return CleanOutput(table=df, meta=meta)

    def clean_all(self, user_id: str, save: bool = False) -> List[CleanOutput]:
        """
        Clean up raw dataset for a single user
        """
        dates = self.uid2dt[user_id]
        it = tqdm(dates, desc=f'Cleaning up raw dataset for user {pl.i(user_id)}', unit='date')
        ret = []
        for d in it:
            it.set_postfix(date=pl.i(d))
            fnm = os_join(user_id, f'{d}.csv')
            ret.append(self.clean_single(fnm, save))
        return ret


if __name__ == '__main__':
    # env_ = 'dev'
    env_ = 'prod'

    dnm = 'raw, 2022-08-10_09-57-34'
    dc = DataCleaner(dataset_path=os_join(u.dataset_path, dnm), verbose=False, root_name=ROOT_HIERARCHY_NAME)

    def single():
        dc.verbose = True
        user_id = dc.user_ids[1]
        mic(user_id)
        # date = dc.uid2dt[user_id][-1]
        date = '2021-05-07'
        # date = '2021-05-21'
        # date = '2022-07-25'
        # date = '2021-09-04'
        # date = '2022-03-30'
        # date = '2020-10-07'
        fnm = os_join(user_id, f'{date}.csv')
        # sv = False
        sv = True
        out = dc.clean_single(data_path=fnm, save=sv)
        df, meta = out.table, out.meta
        mic(df, meta)
    # single()

    def run():
        # dc.clean_all(user_id=user_id, save=True)
        uids = get_user_ids(split=env_)
        for i in uids:
            lst = dc.clean_all(i, save=True)
            mic(i, len(lst))
    run()
