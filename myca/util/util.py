from stefutil import *
from myca.util._data_path import BASE_PATH, PROJ_DIR, PKG_NM, DSET_DIR, MODEL_DIR

__all__ = ['u']

u = StefUtil(
    base_path=BASE_PATH, project_dir=PROJ_DIR, package_name=PKG_NM, dataset_dir=DSET_DIR, model_dir=MODEL_DIR
)
