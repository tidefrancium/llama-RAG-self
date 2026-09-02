from typing import List
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import Document
from llama_index.core.embeddings import BaseEmbedding
from config.settings import SEMANTIC_BUFFER_SIZE, SEMANTIC_BREAKPOINT_THRESHOLD


def get_semantic_splitter(embed_model: BaseEmbedding) -> SemanticSplitterNodeParser:
    """
    获取LlamaIndex原生语义分块器 SemanticSplitterNodeParser
    原理：句子转为向量，计算相邻句子相似度；相似度断崖下跌处作为分割点，实现语义切分
    :param embed_model: 传入embedding模型(OllamaEmbeddings等)
    :return: 语义切片器对象
    """
    splitter = SemanticSplitterNodeParser(
        buffer_size=SEMANTIC_BUFFER_SIZE,
        breakpoint_percentile_threshold=SEMANTIC_BREAKPOINT_THRESHOLD,
        embed_model=embed_model
    )
    return splitter


def semantic_chunk_documents(documents: List[Document], embed_model: BaseEmbedding):
    """
    对外接口：对一批Document执行语义分块，返回Node列表
    :param documents: data_loader加载出来的Document列表
    :param embed_model: embedding模型实例
    :return: list[BaseNode]
    """
    if not documents:
        return []
    splitter = get_semantic_splitter(embed_model)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"✅语义分块完成，生成节点数量：{len(nodes)}")
    return nodes
