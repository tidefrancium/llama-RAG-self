import os
import pymupdf
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from config.settings import (
    RAW_TXT_PATH,
    CHUNK_MAX_CHAR,
    CHUNK_STEP,
    FILE_MD5_CACHE_PATH
)
from utils.file_md5 import scan_all_file_md5
from utils.cache_tool import load_pkl_cache, save_pkl_cache

TXT_MD_SUFFIX = {".txt", ".md"}
PDF_SUFFIX = {".pdf"}
ALL_SUPPORT_SUFFIX = TXT_MD_SUFFIX.union(PDF_SUFFIX)

def get_default_splitter() -> SentenceSplitter:
    return SentenceSplitter(
        chunk_size=CHUNK_MAX_CHAR,
        chunk_overlap=CHUNK_STEP
    )

def load_txt_md_files(file_paths: list[str]) -> list[Document]:
    if not file_paths:
        return []
    reader = SimpleDirectoryReader(input_files=file_paths)
    docs = reader.load_data()
    print(f"加载 txt/md 文件数量：{len(docs)}")
    return docs

def load_pdf_files(file_paths: list[str]) -> list[Document]:
    docs = []
    for fp in file_paths:
        try:
            doc_text = ""
            with pymupdf.open(fp) as pdf_doc:
                for page in pdf_doc:
                    page_text = page.get_text(sort=True).strip()
                    if not page_text:
                        continue
                    doc_text += page_text + "\n\n"
            full_text = doc_text.strip()
            if full_text.startswith("%PDF"):
                print(f"⚠️ 文件 {fp} 检测到PDF二进制损坏头，跳过")
                continue
            if not full_text:
                print(f"⚠️ 文件 {fp} 无可读文本，跳过")
                continue
            doc = Document(
                text=doc_text,
                metadata={
                    "file_path": fp,
                    "file_type": "pdf"
                }
            )
            docs.append(doc)
        except Exception as e:
            print(f"PDF读取失败 {fp}，错误：{str(e)}")
    print(f"加载 pdf 文件数量：{len(docs)}")
    return docs

def split_files_by_type(file_path_list: list[str]):
    txt_md_list = []
    pdf_list = []
    for fp in file_path_list:
        ext = os.path.splitext(fp)[-1].lower()
        if ext in TXT_MD_SUFFIX:
            txt_md_list.append(fp)
        elif ext in PDF_SUFFIX:
            pdf_list.append(fp)
    return txt_md_list, pdf_list

def load_separated_docs(file_path_list: list[str]) -> list[Document]:
    txt_md_paths, pdf_paths = split_files_by_type(file_path_list)
    doc_list = []
    doc_list.extend(load_txt_md_files(txt_md_paths))
    doc_list.extend(load_pdf_files(pdf_paths))
    return doc_list

def load_incremental_documents() -> list[Document]:
    current_md5_map = scan_all_file_md5(RAW_TXT_PATH)
    old_md5_map = load_pkl_cache(FILE_MD5_CACHE_PATH, default={})
    changed_files = []
    for abs_fp, md5_val in current_md5_map.items():
        if old_md5_map.get(abs_fp, "") != md5_val:
            changed_files.append(abs_fp)
    if not changed_files:
        print("无新增/修改文档，无需导入")
        return []
    print(f"检测变更文件总数：{len(changed_files)}")
    documents = load_separated_docs(changed_files)
    save_pkl_cache(current_md5_map, FILE_MD5_CACHE_PATH)
    return documents

def load_all_documents_full() -> list[Document]:
    all_file_paths = []
    for root, _, files in os.walk(RAW_TXT_PATH):
        for fname in files:
            ext = os.path.splitext(fname)[-1].lower()
            if ext in ALL_SUPPORT_SUFFIX:
                full_fp = os.path.abspath(os.path.join(root, fname))
                all_file_paths.append(full_fp)
    return load_separated_docs(all_file_paths)

# 仅保留唯一标准分块入口，删除重名无参函数
def split_docs_into_nodes(documents: list[Document], semantic_mode: bool = False, embed_model=None):
    if not documents:
        return []
    if semantic_mode:
        if embed_model is None:
            raise ValueError("语义分块模式下必须传入embed_model！")
        from data.semantic_chunk import semantic_chunk_documents
        nodes = semantic_chunk_documents(documents, embed_model)
    else:
        splitter = get_default_splitter()
        nodes = splitter.get_nodes_from_documents(documents)
    # 新增：每个节点绑定完整原始chunk文本，用于检索后恢复长段落
    for n in nodes:
        n.metadata["origin_full_chunk"] = n.text.strip()
    print(f"✅传统切片完成，生成节点：{len(nodes)}")
    return nodes


# 移除废弃调试函数 read_single_file，避免误调用单页拆分PDF

if __name__ == "__main__":
    docs = load_incremental_documents()
    nodes = split_docs_into_nodes(docs)
