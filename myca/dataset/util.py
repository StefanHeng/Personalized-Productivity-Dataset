import os
import json
import math
import datetime
from typing import List, Dict

from myca.util import *

__all__ = ['AdjList', 'is_nan', 'str_time2time', 'get_user_ids']


AdjList = Dict[str, List[str]]


def is_nan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


def str_time2time(t: str) -> datetime:
    try:
        return datetime.datetime.strptime(t, '%Y-%m-%dT%H:%M:%S.%f')
    except ValueError:
        return datetime.datetime.strptime(t, '%Y-%m-%dT%H:%M:%S')


def get_user_ids(
        path: str = os.path.join(u.proj_path, 'auth', 'myca', 'user-ids.json'), split: str = 'dev'
) -> List[str]:
    with open(path, 'r') as f:
        user_ids = json.load(f)[split]
        # keeping the prefix still works, but not friendly to file system
        return [i.removeprefix('urn:uuid:') for i in user_ids]
