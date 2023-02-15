"""
Visualize the dataset statistics
"""

import json
import glob
import statistics as stats
from os.path import join as os_join
from typing import List, Tuple, Dict, Union, Any
from collections import Counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from stefutil import *
from myca.util import *
from myca.dataset.clean import ROOT_HIERARCHY_NAME, path2root

# Not every action item has a root category, some entries are at the root-level itself,
# assign them to a `None` category, see `dist_by_root_category`
# NONE_CATEGORY = '__NONE__'
NONE_CATEGORY = '<NONE>'


class MycaVisualizer:
    """
    Plots how data distribution changes over time
    """

    # See `myca.dataset.raw`
    def __init__(
            self, dataset_path: str = None,
            # Inclusive start & end
            start_date: str = '2020-10-01', end_date: str = '2022-08-01', interval: str = '3mo'
    ):
        # assert interval == '3mo'
        ca.check_mismatch('Plot Time Interval', interval, ['1mo', '3mo'])
        self.interval = interval
        self.start_date, self.end_date = start_date, end_date
        self.start_date_t, self.end_date_t = pd.Timestamp(start_date), pd.Timestamp(end_date)

        self.dataset_path = dataset_path
        self.user_ids = [stem(f) for f in sorted(glob.iglob(os_join(self.dataset_path, '*.csv')))]

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

        self.plt_colors = sns.color_palette(palette='husl', n_colors=len(self.user_ids) + 4)

    def _uid2plot_meta(self, uid: str, plot_kind: str = 'n_label', levels=None) -> List[Dict[str, float]]:
        """
        :return: Plot metadata, indexed by time interval
        """
        ca.check_mismatch('Plot Kind', plot_kind, ['n_label', 'n_hier_ch'])

        if plot_kind == 'n_label':
            ret = [[] for _ in self.itv_edges]
            for dates, d in self.hierarchies[uid].items():
                dates = json.loads(dates)
                hier = {int(k): v for k, v in d['hierarchy_graph'].items()}
                vocab = {int(k): v for k, v in d['vocabulary'].items()}
                root_idx = [k for k, v in vocab.items() if v == ROOT_HIERARCHY_NAME]
                assert len(root_idx) == 1  # sanity check
                root_idx = root_idx[0]

                if levels == 'all':
                    n_lb = len(hier) - 1  # exclude root
                else:
                    if levels == 0:  # root labels
                        n_lb = len(hier[root_idx])
                    else:
                        # root is arbitrary, root-level children, should be 0 depth, not 1 in most trees/graphs
                        lb2depth = {lb: len(path2root(graph=hier, root=root_idx, target=lb)) - 2 for lb in hier.keys()}
                        n_lb = len([lb for lb, p_len in lb2depth.items() if p_len == levels])

                d_st, d_ed = dates[0], dates[-1]
                # If the hierarchy stays the same across multiple intervals, it will be counted multiple times
                for i_, (e_s, e_e) in enumerate(self.itv_edges):
                    if pd.Timestamp(d_st) < e_e and pd.Timestamp(d_ed) >= e_s:
                        ret[i_].append(n_lb)
            return [dict(mu=np.mean(n), sig=stats.stdev(n), mi=min(n), ma=max(n)) for n in ret]

    def _uid2root_cat_dist_meta(self, uid: str) -> Tuple[List[Dict[str, float]], List[str]]:
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
        assert NONE_CATEGORY not in root_cats
        root_cats.append(NONE_CATEGORY)

        df.date = df.date.apply(lambda x: pd.Timestamp(x))
        for i, (e_s, e_e) in enumerate(self.itv_edges):
            flag = (e_s <= df.date) & (df.date < e_e)
            lbs = df[flag].labels.apply(lambda x: x[0] if len(x) == 1 else NONE_CATEGORY)
            ret[i] = c = Counter(lbs)
            assert len(lbs) == c.total()  # sanity check
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
        for i, uid in enumerate(self.user_ids):
            df = pd.DataFrame(self._uid2plot_meta(uid=uid, plot_kind='n_label', levels=levels))

            u_lb = f'User-{i+1}-{uid[:4]}'
            c = self.plt_colors[i]
            args = LN_KWARGS | dict(ms=2, c=c, lw=0.7)
            plt.plot(self._interval_2_plot_centers(), df.mu, label=u_lb, **args)
            assert len(df.sig) + 1 == len(x_bounds)
            for j, row in df.iterrows():
                j: int
                # 1std range as box, min and max as horizontal lines
                plt.fill_between(x_bounds[j:j+2], row.mu - row.sig, row.mu + row.sig, facecolor=c, alpha=0.1)
                args.update(dict(marker=None, lw=0.3))
                plt.plot(x_bounds[j:j+2], [row.mi, row.mi], alpha=0.5, **args)
                plt.plot(x_bounds[j:j+2], [row.ma, row.ma], alpha=0.5, **args)

        mi, ma = ax.get_ylim()
        ax.set_ylim([0, ma])  # y-axis starts from 0

        args = dict(lw=0.4, color=self.plt_colors[-1], alpha=0.5)
        plt.vlines(x=x_bounds, ymin=0, ymax=ma, label='Time Interval Boundaries', **args)
        ax.set_xlim([x_bounds[0], x_bounds[-1]])

        if levels == 'all':
            lvl_desc = 'All Levels'
        else:
            lvl_desc = 'Root Labels' if levels == 0 else f'{ordinal(levels+1)} Level'
        plt.title(f'#Categories over time on {lvl_desc}')
        plt.xlabel('Date')
        plt.ylabel('#Categoriy')
        plt.legend()
        plt.show()

    def dist_by_root_category(self, category_order: str = 'count', count_coverage_ratio: int = 0.95):
        """
        Stacked histogram on #action items added for each root category overtime

        :param category_order: Ordering of stacked categories/legend, one of [`count`, `time`]
        :param count_coverage_ratio: The ratio of total #actions to be covered by the top occurring categories
        """
        ca.check_mismatch('Category Plot Order', category_order, ['count', 'time'])
        order_by_count = category_order == 'count'

        for i, uid in enumerate(self.user_ids):
            fig = plt.figure()
            ax = plt.gca()
            # TODO: If using histogram lib, the time interval has to be the same?
            meta, root_cats = self._uid2root_cat_dist_meta(uid=uid)

            palette = None
            if order_by_count:
                total_counts = Counter()
                for m in meta:
                    total_counts.update(m)
                mic(total_counts)
                root_cats_ = [k for k, _ in total_counts.most_common()]
                assert set(root_cats_) == set(root_cats)  # sanity check
                root_cats = root_cats_
                mic(root_cats)

                # Make hues more differentiable for the highest occurring categories
                cov_ratio = np.cumsum([total_counts[k] for k in root_cats]) / sum(total_counts.values())
                mic(cov_ratio)
                n_larger_cat = int(np.searchsorted(cov_ratio, count_coverage_ratio, side='right'))
                mic(n_larger_cat)
                palette = sns.color_palette(palette='husl', n_colors=n_larger_cat)
                # a different palette to differentiate
                palette += sns.color_palette(palette='RdYlBu', n_colors=len(root_cats) - n_larger_cat)
                # raise NotImplementedError

            # Order the categories, instead of
            # df = pd.DataFrame([{k: m.get(k, 0) for k in root_cats} for m in meta])  # So that dataframe is ordered
            rows = []
            # Assign unique date-interval category to each row to enforce bins in histogram
            date_cats = [f'{i:>2}_{s.date()}_{e.date()}' for i, (s, e) in enumerate(self.itv_edges)]
            date_cats_internal = [f'{i:>2}' for i in range(len(self.itv_edges))]
            for j, d_c in enumerate(date_cats_internal):
                rows.extend([{'cat': d_c, 'root_cat': k, 'count': meta[j].get(k, 0)} for k in root_cats])
            # mic(df)
            df = pd.DataFrame(rows)
            # mic(df.columns, len(df.columns))
            # df['cat'] = cats = [f'{i:>2}_{s.date()}_{e.date()}' for i, (s, e) in enumerate(self.itv_edges)]
            df_col2cat_col(df, 'cat', date_cats_internal)
            mic(df, root_cats)
            ax = sns.histplot(
                data=df, x='cat', weights='count', bins=len(date_cats),
                discrete=True, multiple='stack', hue='root_cat', palette=palette,
                # stat='percent',
                ax=ax
            )
            # fig.legend(loc='outside right upper')
            sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1))
            u_lb = f'User-{i+1}-{uid[:4]}'
            ax.get_legend().set_title(f'{u_lb}\'s Root Categories, ordered by {category_order.capitalize()}')
            plt.show()

            raise NotImplementedError

    def _interval_2_plot_centers(self) -> List[pd.Timestamp]:
        return [pd.Timestamp((s.value+e.value)/2) for (s, e) in self.itv_edges]

    def _get_interval_boundaries(self) -> List[pd.Timestamp]:
        return [self.itv_edges[0][0]] + [e for (s, e) in self.itv_edges]


if __name__ == '__main__':
    mic.output_width = 256

    dnm = '23-02-10_Aggregated-Dataset'
    mv = MycaVisualizer(dataset_path=os_join(u.dset_path, dnm), interval='1mo')

    def check_n_label():
        # mv.n_label(levels='all')
        mv.n_label(levels=0)
        # mv.n_label(levels=1)
        # mv.n_label(levels=2)
        # mv.n_label(levels=3)
        # mv.n_label(levels=4)  # The deepest level
    # check_n_label()

    def check_root_cat_dist():
        mv.dist_by_root_category()
    check_root_cat_dist()
