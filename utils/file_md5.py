import hashlib
import os

def get_file_md5(file_path: str) -> str:
    """计算单个文件md5，用于判断文件内容是否发生变更"""
    hash_obj = hashlib.md5()
    with open(file_path, "rb") as f:
        # 分块读取大文件，防止一次性读入内存
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def scan_all_file_md5(root_dir: str) -> dict:
    """
    遍历目录，扫描全部支持文档，返回 {文件绝对路径:md5字符串}
    """
    file_md5_map = {}
    allow_suffix = {".txt", ".md", ".pdf"}
    for root, _, files in os.walk(root_dir):
        for filename in files:
            ext = os.path.splitext(filename)[-1].lower()
            if ext not in allow_suffix:
                continue
            abs_file = os.path.abspath(os.path.join(root, filename))
            file_md5_map[abs_file] = get_file_md5(abs_file)
    return file_md5_map
