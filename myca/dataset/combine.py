"""
Aggregate for each user, across dates, into collection of unique action entries
A finalized version
"""

from os.path import join as os_join

from stefutil import *
from myca.util import *


class DataAggregator:
    def __init__(
            self, dataset_path: str, output_path: str = os_join(u.dset_path, f'aggregated, {now(for_path=True)}'),
            verbose: bool = True
    ):
        pass
