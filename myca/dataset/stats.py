"""
Visualize the dataset statistics
"""

import json
import glob
import statistics as stats
from os.path import join as os_join
from typing import List, Tuple, Dict, Union, Any
from dataclasses import dataclass
from collections import Counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import zss
from zss import Node
from tqdm.auto import tqdm

from stefutil import *
from myca.util import *
from myca.dataset.clean import ROOT_HIERARCHY_NAME, path2root

# Not every action item has a root category, some entries are at the root-level itself,
# assign them to a `None` category, see `dist_by_root_category`
# NONE_CATEGORY = '__NONE__'
NONE_CATEGORY = '<NONE>'


@dataclass
class HierarchyMeta:
    tree: Dict[int, List[int]] = None
    root_node: int = None

    def to_str_nodes(self) -> Dict[str, Any]:
        return dict(
            tree={str(k): [str(v_) for v_ in v] for k, v in self.tree.items()}, root_node=str(self.root_node)
        )


logger = get_logger('Myca Visualizer')


class MycaVisualizer:
    """
    Plots how data distribution changes over time
    """

    # See `myca.dataset.raw`
    def __init__(
            self, dataset_path: str = None,
            # Inclusive start & end
            start_date: str = '2020-10-01', end_date: str = '2022-08-01', interval: str = '3mo',
            show_title: bool = True, save: bool = False
    ):
        # assert interval == '3mo'
        ca.check_mismatch('Plot Time Interval', interval, ['1mo', '3mo'])
        self.interval = interval
        self.start_date, self.end_date = start_date, end_date
        self.start_date_t, self.end_date_t = pd.Timestamp(start_date), pd.Timestamp(end_date)

        self.dataset_path = dataset_path
        self.user_ids = [stem(f) for f in sorted(glob.iglob(os_join(self.dataset_path, '*.csv')))]
        # sanity check a shorter code is still unique
        assert len(set([us[:4] for us in self.user_ids])) == len(self.user_ids)

        def load_hier(uid: str):
            path_hier = os_join(self.dataset_path, f'{uid}.json')
            with open(path_hier, 'r') as f:
                return json.load(f)
        self.hierarchies = {uid: load_hier(uid) for uid in self.user_ids}

        self.itv_edges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []  # Inclusive start, exclusive end
        f = '3MS' if interval == '3mo' else '1MS'
        m_starts = pd.date_range(start=start_date[:7], end=end_date, freq=f)  # TODO: generalize
        assert m_starts[0] <= self.start_date_t

        if m_starts[0] < self.start_date_t:
            m_starts[0] = self.start_date_t
        for i, t in enumerate(m_starts[:-1]):
            s, e = t, m_starts[i+1]
            if e > self.end_date_t:
                e = self.end_date_t
            self.itv_edges.append((s, e))
        if m_starts[-1] < self.end_date_t:
            self.itv_edges.append((m_starts[-1], self.end_date_t))
        s, e = self.itv_edges[-1]
        # TODO: better ways to plot than merging the last group?
        self.itv_edges[-1] = (s, e+pd.Timedelta(days=1))  # Inclusive end

        self.plt_colors_by_user = sns.color_palette(palette='husl', n_colors=len(self.user_ids) + 4)

        self.show_title = show_title
        self.save = save  # Plots are saved to disk instead of shown

    def _uid2n_label_meta(self, uid: str, levels=None) -> List[Dict[str, float]]:
        """
        :return: Plot metadata, indexed by time interval
        """
        ret = [[] for _ in self.itv_edges]
        for dates, d in self.hierarchies[uid].items():
            dates = json.loads(dates)
            out = MycaVisualizer.user_hierarchy_meta2tree(d)
            hier, root_idx = out.tree, out.root_node

            if levels == 'all':
                n_lb = len(hier) - 1  # exclude root
            else:
                if levels == 0:  # root labels
                    n_lb = len(hier[root_idx])
                else:
                    # root is arbitrary, root-level children, should be 0 depth, not 1 in most trees/graphs
                    lb2depth = {lb: len(path2root(graph=hier, root=root_idx, target=lb)) - 2 for lb in hier.keys()}
                    n_lb = len([lb for lb, p_len in lb2depth.items() if p_len == levels])

            d_st, d_ed = pd.Timestamp(dates[0]), pd.Timestamp(dates[-1])
            # If the hierarchy stays the same across multiple intervals, it will be counted multiple times
            for i_, (e_s, e_e) in enumerate(self.itv_edges):
                if d_st < e_e and e_s <= d_ed:  # Exclusive end for time interval
                    ret[i_].append(n_lb)
        return [dict(mu=np.mean(n), sig=stats.stdev(n), mi=min(n), ma=max(n)) for n in ret]

    @staticmethod
    def user_hierarchy_meta2tree(d: Dict[str, Any]) -> HierarchyMeta:
        hier = {int(k): [int(v_) for v_ in v] for k, v in d['hierarchy_graph'].items()}
        vocab = {int(k): v for k, v in d['vocabulary'].items()}
        root_idx = [k for k, v in vocab.items() if v == ROOT_HIERARCHY_NAME]
        assert len(root_idx) == 1  # sanity check
        root_idx = root_idx[0]
        return HierarchyMeta(tree=hier, root_node=root_idx)

    def _uid2n_hier_ch_meta(self, uid: str, change_kind: str) -> Union[List[int], List[Dict[str, int]]]:
        whole_counts = [0 for _ in self.itv_edges]

        lst_dates_by_itv: List[List[str]] = [[] for _ in self.itv_edges]
        for k_dates in self.hierarchies[uid].keys():
            dates = json.loads(k_dates)
            d_st, d_ed = pd.Timestamp(dates[0]), pd.Timestamp(dates[-1])
            for i_, (e_s, e_e) in enumerate(self.itv_edges):  # Get #unique hierarchies
                if d_st < e_e and e_s <= d_ed:
                    whole_counts[i_] += 1
                    lst_dates_by_itv[i_].append(dates)
        if change_kind == 'single':
            d_by_itv = [Counter() for _ in self.itv_edges]
            op2str_op = {zss.Operation.insert: 'add', zss.Operation.remove: 'delete', zss.Operation.update: 'rename'}
            for i_, lst_dates in enumerate(lst_dates_by_itv):
                it = tqdm(zip(lst_dates[:-1], lst_dates[1:]), desc='Counting edits', total=len(lst_dates)-1)
                for d_prev, d_curr in it:
                    d_hier = self.hierarchies[uid]
                    out_prev = MycaVisualizer.user_hierarchy_meta2tree(d_hier[json.dumps(d_prev)])
                    out_curr = MycaVisualizer.user_hierarchy_meta2tree(d_hier[json.dumps(d_curr)])
                    tr_prev = MycaVisualizer._hierarchy2zss_tree(out_prev)
                    tr_curr = MycaVisualizer._hierarchy2zss_tree(out_curr)

                    # Since the hierarchy is digital anyway, any kind of edit has the same cost
                    cost, ops = zss.distance(
                        tr_prev, tr_curr, get_children=Node.get_children, return_operations=True,
                        insert_cost=(lambda x: 1), remove_cost=(lambda x: 1), update_cost=(lambda x, y: 1)
                    )
                    # `match` looks like internal var for zss DP implementation'
                    ops: List[zss.Operation] = [op for op in ops if op != zss.Operation(zss.Operation.match)]
                    ops_s: List[str] = [op2str_op[op.type] for op in ops]
                    if cost != len(ops):
                        mic(out_prev.tree, out_curr.tree)
                        mic(cost, ops, len(ops))
                        # raise NotImplementedError
                        logger.warning(f'Cost does not match #ops: {pl.i(cost)} != {pl.i(len(ops))}')
                    # assert cost == len(ops)  # sanity check
                    # mic(ops_s)
                    d_by_itv[i_].update(ops_s)
                    # mic(d_by_itv[i_])
                    # raise NotImplementedError
            return [dict(d) for d in d_by_itv]
        else:  # `whole`
            return [c-1 for c in whole_counts]  # For #change

    @staticmethod
    def _hierarchy2zss_tree(hierarchy: HierarchyMeta) -> Node:
        """
        :param hierarchy: Adjacency list of category hierarchy read in from file
        """
        # Convert all notes to str for `zss`
        tree = {str(k): [str(v_) for v_ in v] for k, v in hierarchy.tree.items()}
        str_node2node: Dict[str, Node] = {}
        # mic(tree)
        for node, children in tree.items():
            n = str_node2node[node] = str_node2node.get(node, Node(node))  # Creates the node if not exists
            for child in children:
                c = str_node2node[child] = str_node2node.get(child, Node(child))
                n.addkid(c)

        sanity_check = False
        if sanity_check:
            tree_recon = dict()
            n_root = str_node2node[str(hierarchy.root_node)]
            to_visit = [n_root]
            while len(to_visit) > 0:
                n = to_visit.pop()
                tree_recon[n.label] = [c.label for c in n.children]
                to_visit.extend(n.children)
            assert tree_recon == tree
        return str_node2node[str(hierarchy.root_node)]

    def _uid2root_cat_dist_meta(self, uid: str, with_null_category: bool) -> Tuple[List[Dict[str, float]], List[str]]:
        """
        :return: 2-tuple of (category #entry counts, list of category ordered by time)
        """
        # TODO: only consider unique (text, label) pairs?
        ret: List[Any] = [None for _ in self.itv_edges]

        df = pd.read_csv(os_join(self.dataset_path, f'{uid}.csv'))
        df = df[df.labels.notnull()]
        df.labels = lbs = df.labels.apply(lambda x: json.loads(x))

        def get_all_root_labels(labels: pd.Series) -> pd.Series:
            return labels[labels.apply(lambda x: len(x) == 1)].apply(lambda x: x[0])
        root_cats = list(get_all_root_labels(lbs).unique())
        # The original df is sorted by time, hence sorted after `unique`

        df.date = df.date.apply(lambda x: pd.Timestamp(x))
        n_null = df.labels.notnull()
        append_null = False
        for i, (e_s, e_e) in enumerate(self.itv_edges):
            flag = (e_s <= df.date) & (df.date < e_e)
            if with_null_category:
                lbs = df[flag].labels.apply(lambda x: x[0] if len(x) else NONE_CATEGORY)
            else:
                flag &= n_null
                lbs = df[flag].labels.apply(lambda x: x[0])
            ret[i] = c = Counter(lbs)
            if NONE_CATEGORY in c:
                append_null = True
            assert len(lbs) == c.total()  # sanity check

        if with_null_category and append_null:
            assert NONE_CATEGORY not in root_cats
            root_cats.append(NONE_CATEGORY)
        assert sum(sum(c.values()) for c in ret) == len(df)  # sanity check
        return ret, root_cats

    def n_label(self, levels: Union[str, int] = 'all'):
        """
        # unique label/category
        """
        assert levels == 'all' or isinstance(levels, int)  # sanity check

        plt.figure()
        ax = plt.gca()
        x_bounds = self._get_interval_boundaries()
        x_centers = self._interval_2_plot_centers()
        for i, uid in enumerate(self.user_ids):
            df = pd.DataFrame(self._uid2n_label_meta(uid=uid, levels=levels))

            u_lb = user_id2str(user_id=uid, index=i)
            c = self.plt_colors_by_user[i]
            args = LN_KWARGS | dict(ms=2, c=c, lw=0.7)
            plt.plot(x_centers, df.mu, label=u_lb, **args)
            assert len(df.sig) + 1 == len(x_bounds)
            for j, row in df.iterrows():
                j: int
                # 1std range as box, min and max as horizontal lines
                plt.fill_between(x_bounds[j:j+2], row.mu - row.sig, row.mu + row.sig, facecolor=c, alpha=0.1)
                args.update(dict(marker=None, lw=0.3))
                plt.plot(x_bounds[j:j+2], [row.mi, row.mi], alpha=0.5, **args)
                plt.plot(x_bounds[j:j+2], [row.ma, row.ma], alpha=0.5, **args)

        self._setup_plot_box(ax=ax)

        if levels == 'all':
            lvl_pref = 'All-level'
        else:
            lvl_pref = 'Root-level' if levels == 0 else f'{ordinal(levels+1)}-level'
        title = f'#{lvl_pref}-Categories over time'
        if self.show_title:
            plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('#Category')
        plt.legend()
        if self.save:
            save_fig(title)
        else:
            plt.show()

    def dist_by_root_category(
            self, category_order: str = 'count', count_coverage_ratio: int = 0.9, single_user: Union[int, str] = None,
            with_null_category: bool = False
    ):
        """
        Stacked histogram on #action items added for each root category overtime

        :param category_order: Ordering of stacked categories/legend, one of [`count`, `time`]
        :param count_coverage_ratio: The ratio of total #actions to be covered by the top occurring categories
        :param single_user: If specified, only plot for the user with the given index or uid
        :param with_null_category: If True, include the `None` category in the plot
        """
        ca.check_mismatch('Category Plot Order', category_order, ['count', 'time'])
        order_by_count = category_order == 'count'

        is_single_user = single_user is not None

        def plot_single(uid_, ax=None, user_idx: int = None):
            ax = ax or plt.gca()
            # TODO: If using histogram lib, the time interval has to be the same?
            meta, root_cats = self._uid2root_cat_dist_meta(uid=uid_, with_null_category=with_null_category)

            palette = None
            if order_by_count:
                total_counts = Counter()
                for m in meta:
                    total_counts.update(m)
                root_cats_ = [k for k, _ in total_counts.most_common()]
                assert set(root_cats_) == set(root_cats)  # sanity check
                root_cats = root_cats_

                # Make hues more differentiable for the highest occurring categories
                cov_ratio = np.cumsum([total_counts[k] for k in root_cats]) / sum(total_counts.values())
                n_larger_cat = int(np.searchsorted(cov_ratio, count_coverage_ratio, side='right'))
                palette = sns.color_palette(palette='husl', n_colors=n_larger_cat)
                # a different palette to differentiate
                palette += sns.color_palette(palette='RdYlBu', n_colors=len(root_cats) - n_larger_cat)

            rows = []
            # Assign unique date-interval category to each row to enforce bins in histogram
            date_cats_internal = [f'{i:>2}' for i in range(len(self.itv_edges))]
            for j_, d_c in enumerate(date_cats_internal):
                rows.extend([{'cat': d_c, 'root_cat': k, 'count': meta[j_].get(k, 0)} for k in root_cats])
            df = pd.DataFrame(rows)
            df_col2cat_col(df, 'cat', date_cats_internal)
            ax = sns.histplot(
                data=df, x='cat', weights='count', bins=len(date_cats_internal),
                discrete=True, multiple='stack', hue='root_cat', palette=palette,
                ax=ax  # TODO: normalize w.r.t. each time interval?
            )

            # Move tick labels from bin centers to bin edges, map to the time space
            xt = ax.get_xticks()
            xt = [t-0.5 for t in xt]
            xt.append(xt[-1] + 1)
            edges = [e.strftime('%Y-%m-%d') for e in self._get_interval_boundaries()]  # Short date
            assert len(edges) == len(xt)  # sanity check

            interval = 3  # Filter s.t. labels don't overlap

            xt = [e for i, e in enumerate(xt) if i % interval == 0]
            edges = [e for i, e in enumerate(edges) if i % interval == 0]
            ax.set_xticks(xt, labels=edges)
            plt.xticks(rotation=45)
            ax.set_xlim([xt[0], xt[-1]])  # snap to the first and last bins

            plt.xlabel('date')
            sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1))
            u_lb = user_id2str(user_id=uid_, index=user_idx)
            ax.title.set_text(u_lb)
            lg = ax.get_legend()
            lg.set_title(f'Root Categories ordered by {category_order.capitalize()}')
            root_cats_san = [sanitize_str(c) for c in root_cats]
            txts = lg.get_texts()
            assert len(txts) == len(root_cats_san)  # sanity check
            for t_obj, t in zip(txts, root_cats_san):
                t_obj.set_text(t)

        title = 'Distribution of #Entry per root category over time'
        if is_single_user:
            if isinstance(single_user, int):
                assert 0 <= single_user < len(self.user_ids)
                u_idx = single_user
                single_user = self.user_ids[single_user]
            else:
                assert single_user in self.user_ids
                u_idx = self.user_ids.index(single_user)
            plt.figure()
            plot_single(single_user, ax=None, user_idx=u_idx)
        else:
            assert len(self.user_ids) == 4  # TODO: generalize
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
            for j, uid in enumerate(self.user_ids):
                plot_single(uid, ax=axs[j//2, j % 2], user_idx=j)
            raise NotImplementedError('Issues; Legend overlaps, plot too small')

        if self.show_title:
            plt.suptitle(title)
        if self.save:
            if single_user:
                title = f'{title} for {user_id2str(user_id=single_user, index=u_idx)}'
            save_fig(title)
        else:
            plt.show()

    def n_hierarchy_change(self, change_kind: str = 'single'):
        """
        :param change_kind: The kind of hierarchy change to plot, one of [`single`, `whole`]
            If `single`, plots the 3 specific kinds of tree edits for every single hierarchy change
            If `whole`, plots the total number of hierarchy changes
        """
        plt.figure()
        ax = plt.gca()
        x_centers = self._interval_2_plot_centers()
        for i, uid in enumerate(self.user_ids):
            meta = self._uid2n_hier_ch_meta(uid=uid, change_kind=change_kind)
            if change_kind == 'single':
                raise NotImplementedError('TODO: too many RENAMEs???  Tree edit algorithm wrong?')
                df = pd.DataFrame(meta)
                mic(df)
                for k in df.columns:
                    plt.plot(x_centers, df[k], label=k, **LN_KWARGS)
                break
            else:
                u_lb = user_id2str(user_id=uid, index=i)
                c = self.plt_colors_by_user[i]
                args = LN_KWARGS | dict(ms=2, c=c, lw=0.7)
                plt.plot(x_centers, meta, label=u_lb, **args)
        self._setup_plot_box(ax=ax)

        title = '#Category Hierarchy Shifts over time'
        if self.show_title:
            plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('#Shift')
        plt.legend()
        if self.save:
            save_fig(title)
        else:
            plt.show()

    def _interval_2_plot_centers(self) -> List[pd.Timestamp]:
        return [pd.Timestamp((s.value+e.value)/2) for (s, e) in self.itv_edges]

    def _get_interval_boundaries(self) -> List[pd.Timestamp]:
        return [self.itv_edges[0][0]] + [e for (s, e) in self.itv_edges]

    def _setup_plot_box(self, ax):
        x_bounds = self._get_interval_boundaries()
        mi, ma = ax.get_ylim()
        ax.set_ylim([min(mi, 0), ma])  # y-axis at least starts from 0

        args = dict(lw=0.4, color=self.plt_colors_by_user[-1], alpha=0.5)
        plt.vlines(x=x_bounds, ymin=0, ymax=ma, label='Time Interval Boundaries', **args)
        ax.set_xlim([x_bounds[0], x_bounds[-1]])


if __name__ == '__main__':
    mic.output_width = 256

    dnm = '23-02-10_Aggregated-Dataset'
    mv = MycaVisualizer(dataset_path=os_join(u.dset_path, dnm), interval='1mo', show_title=False, save=False)

    def check_n_label():
        mv.n_label(levels='all')
        mv.n_label(levels=0)
        mv.n_label(levels=1)
        mv.n_label(levels=2)
        mv.n_label(levels=3)
        mv.n_label(levels=4)  # The deepest level
    # check_n_label()

    def check_root_cat_dist():
        for i in range(4):
            mv.dist_by_root_category(single_user=i, with_null_category=True)
            # raise NotImplementedError
    # check_root_cat_dist()

    def check_n_hierarchy_change():
        mv.n_hierarchy_change(change_kind='whole')
    check_n_hierarchy_change()
