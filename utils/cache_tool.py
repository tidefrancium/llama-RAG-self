import pickle
import os

def save_pkl_cache(obj, cache_path: str):
    """把对象序列化保存到pkl缓存文件，自动创建父文件夹"""
    dir_name = os.path.dirname(cache_path)
    os.makedirs(dir_name, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(obj, f)

def load_pkl_cache(cache_path: str, default=None):
    """读取pkl缓存；文件不存在返回default"""
    if not os.path.exists(cache_path):
        return default
    with open(cache_path, "rb") as f:
        return pickle.load(f)
