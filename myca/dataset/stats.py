"""
Visualize the dataset statistics
"""

import json
import glob
import statistics as stats
from os.path import join as os_join
from typing import List, Tuple, Dict, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from stefutil import *
from myca.util import *
from myca.dataset.clean import ROOT_HIERARCHY_NAME, path2root


class MycaVisualizer:
    """
    Plots how data distribution changes over time
    """

    # See `myca.dataset.raw`
    def __init__(
            self, dataset_path: str = None,
            start_date: str = '2020-10-01', end_date: str = '2022-08-01', interval: str = '3mo'
    ):
        assert interval == '3mo'
        self.interval = interval
        self.start_date, self.end_date = start_date, end_date
        self.start_date_t, self.end_date_t = pd.Timestamp(start_date), pd.Timestamp(end_date)

        self.dataset_path = dataset_path
        self.user_ids = [stem(f) for f in sorted(glob.iglob(os_join(self.dataset_path, '*.csv')))]
        mic(self.user_ids)

        def load_hier(uid: str):
            path_hier = os_join(self.dataset_path, f'{uid}.json')
            with open(path_hier, 'r') as f:
                return json.load(f)
        self.hierarchies = {uid: load_hier(uid) for uid in self.user_ids}

        self.itv_edges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []  # Inclusive start, exclusive end
        m_starts = pd.date_range(start=start_date[:7], end=end_date, freq='3MS')  # TODO: generalize
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

        self.plt_colors = sns.color_palette(palette='husl', n_colors=len(self.user_ids) + 4)

    def n_label(self, levels: Union[str, int] = 'all'):
        is_all_level = levels == 'all'
        if not is_all_level:  # sanity check
            assert isinstance(levels, int)

        def uid2n_label(uid: str) -> List[Dict[str, float]]:
            # Get Unique #label for the given time interval, ordered by time
            ret = [[] for _ in self.itv_edges]  # map to time interval by index
            for dates, d in self.hierarchies[uid].items():
                dates = json.loads(dates)
                hier = {int(k): v for k, v in d['hierarchy_graph'].items()}
                vocab = {int(k): v for k, v in d['vocabulary'].items()}
                root_idx = [k for k, v in vocab.items() if v == ROOT_HIERARCHY_NAME]
                assert len(root_idx) == 1  # sanity check
                root_idx = root_idx[0]

                if is_all_level:
                    n_lb = len(hier)-1  # exclude root
                else:
                    if levels == 0:  # root labels
                        n_lb = len(hier[root_idx])
                    else:
                        # root is arbitrary, root-level children, should be 0 depth, not 1 in most trees/graphs
                        lb2depth = {lb: len(path2root(graph=hier, root=root_idx, target=lb)) - 2 for lb in hier.keys()}
                        n_lb = len([lb for lb, p_len in lb2depth.items() if p_len == levels])
                        # if uid == 'dcd924b1-8e9a-4f32-9edf-310626552878':
                        #     mic(lb2depth, n_lb, levels)
                        #     raise NotImplementedError

                d_st, d_ed = dates[0], dates[-1]

                # If the hierarchy stays the same across multiple intervals, it will be counted multiple times
                for i_, (e_s, e_e) in enumerate(self.itv_edges):
                    if pd.Timestamp(d_st) < e_e and pd.Timestamp(d_ed) >= e_s:
                        ret[i_].append(n_lb)

            return [dict(mu=np.mean(n), sig=stats.stdev(n), mi=min(n), ma=max(n)) for n in ret]

        plt.figure()
        ax = plt.gca()
        x_bounds = self._get_interval_boundaries()
        for i, uid_ in enumerate(self.user_ids):
            df = pd.DataFrame(uid2n_label(uid_))

            u_lb = f'User-{i+1}-{uid_[:4]}'
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

    def _interval_2_plot_centers(self) -> List[pd.Timestamp]:
        return [pd.Timestamp((s.value+e.value)/2) for (s, e) in self.itv_edges]

    def _get_interval_boundaries(self) -> List[pd.Timestamp]:
        return [self.itv_edges[0][0]] + [e for (s, e) in self.itv_edges]


if __name__ == '__main__':
    mic.output_width = 256

    dnm = '23-02-10_Aggregated-Dataset'
    mv = MycaVisualizer(dataset_path=os_join(u.dset_path, dnm))
    mv.n_label(levels='all')
    # mv.n_label(levels=0)
    # mv.n_label(levels=1)
    # mv.n_label(levels=2)
    # mv.n_label(levels=3)
    # mv.n_label(levels=4)  # The deepest level
