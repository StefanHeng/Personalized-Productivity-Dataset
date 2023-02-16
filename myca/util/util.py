from stefutil import *
from myca.util._data_path import BASE_PATH, PROJ_DIR, PKG_NM, DSET_DIR, MODEL_DIR

__all__ = ['u', 'save_fig', 'user_id2str']

u = StefUtil(
    base_path=BASE_PATH, project_dir=PROJ_DIR, package_name=PKG_NM, dataset_dir=DSET_DIR, model_dir=MODEL_DIR
)
save_fig = u.save_fig


def user_id2str(user_id: str, index: int = None):
    ret = 'User-'
    if index is not None:
        ret += f'{index+1}-'
    ret += user_id[:4]
    return ret
