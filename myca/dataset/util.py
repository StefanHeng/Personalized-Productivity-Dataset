import os
import json
import math
from typing import List

from myca.util import *

__all__ = ['is_nan', 'get_user_ids']


def is_nan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


def get_user_ids(
        path: str = os.path.join(u.proj_path, 'auth', 'myca', 'user-ids.json'), split: str = 'dev'
) -> List[str]:
    with open(path, 'r') as f:
        user_ids = json.load(f)[split]
        # keeping the prefix still works, but not friendly to file system
        return [i.removeprefix('urn:uuid:') for i in user_ids]
