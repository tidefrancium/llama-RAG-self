import sys
from pathlib import Path
file_full = Path(__file__).resolve()
ROOT_DIR = file_full.parent.parent
sys.path.insert(0, str(ROOT_DIR))
import os
import shutil
import pickle
from datetime import datetime
import gradio as gr
from config.settings import RAW_TXT_PATH
from embedder.vector_embed import build_vector_index, load_exist_vector_index
from generator.generate_answer import rag_full_workflow
from data.data_loder import load_all_documents_full, split_docs_into_nodes

# ===================== 全局状态容器（解决Gradio多线程全局变量丢失BUG） =====================
APP_STATE = {
    "vec_index": None,          # 向量库索引实例（核心，全局共享）
    "all_nodes": None,          # 全部分块节点
    "processed_files": set()    # 已入库文件路径集合
}
# ==========================================================================================

# 持久化已入库文件记录路径，用于启动自动加载
RECORD_PATH = ROOT_DIR / "config" / "processed_files_record.pkl"
DUMP_LOG_PATH = ROOT_DIR / "rag_doc_dump.log"
# LlamaIndex元缓存目录（统一硬编码，消除get_index_cache_dir导入依赖）
INDEX_CACHE_DIR = ROOT_DIR / "index_cache"

# ===================== 蓝‑紫‑深蓝分层CSS 自适应网页 =====================
CUSTOM_CSS = """
/* 全局：自适应，最大宽度限制，小屏幕自动收缩 */
.gradio-container {
    max-width: 92vw !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding: 20px 16px !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: linear-gradient(145deg, #0f1730 0%, #182244 100%) !important;
}
/* 卡片容器：深蓝底色 + 浅蓝紫边框分层，自适应宽度 */
.module-card {
    width: 100% !important;
    border-radius: 16px !important;
    background: rgba(30, 42, 80, 0.45) !important;
    border: 1px solid rgba(140, 160, 255, 0.22) !important;
    box-shadow: 0 6px 24px rgba(20, 30, 70, 0.45) !important;
    margin-bottom: 24px !important;
    padding: 22px 26px !important;
}
/* 标题 */
.main-title {
    margin-bottom: 6px !important;
    font-weight: 600 !important;
    color: #c7d2ff !important;
}
.sub-title {
    color: #94a3e8 !important;
    font-size: 14px !important;
    margin-bottom: 0 !important;
}
/* 按钮分层配色 */
button.primary {
    background: linear-gradient(135deg,#406bff 0%,#7b61ff 100%) !important;
    border: none !important;
    color:#ffffff !important;
    border-radius:10px !important;
    font-weight:500 !important;
}
button.secondary {
    background: linear-gradient(135deg,#364899 0%,#5246aa 100%) !important;
    border: none !important;
    color:#e6eaff !important;
    border-radius:10px !important;
}
button.stop {
    background: linear-gradient(135deg,#993658 0%,#aa4662 100%) !important;
    border: none !important;
    color:#ffffff !important;
    border-radius:10px !important;
}
button:hover {
    filter: brightness(1.15);
    transform: translateY(-1px);
    transition: all 0.22s ease;
}
/* 状态文本框 */
.status-box textarea {
    background: rgba(18, 26, 58, 0.7) !important;
    border: 1px solid rgba(120,140,230,0.25) !important;
    border-radius:12px !important;
    color:#c2cfff !important;
}
/* Chatbot聊天气泡 蓝紫分层 */
.chatbot {
    background: rgba(22, 32, 64, 0.55) !important;
    border:1px solid rgba(130,150,240,0.2) !important;
    border-radius:14px !important;
}
.chatbot .user {
    background: linear-gradient(135deg,#3b68ff 0%,#7258ee 100%) !important;
    color:#fff !important;
    border-radius: 14px 14px 4px 14px !important;
}
.chatbot .bot {
    background: rgba(60, 50, 110, 0.4) !important;
    color:#e2e8ff !important;
    border:1px solid rgba(140,130,220,0.22) !important;
    border-radius:14px 14px 14px 4px !important;
}
/* 输入框 */
textarea[type="text"], textarea {
    background: rgba(24, 34, 70, 0.6) !important;
    border: 1px solid rgba(130,150,240,0.24) !important;
    color:#e6ebff !important;
    border-radius:12px !important;
}
/* 折叠面板 检索原文 */
.accordion {
    background: rgba(28, 38, 78, 0.45) !important;
    border:1px solid rgba(130,150,240,0.22) !important;
    border-radius:12px !important;
}
.accordion textarea {
    background:rgba(16, 24, 54,0.65)!important;
    color:#c4cdff !important;
}
"""

def save_process_record(file_set: set):
    """持久化保存已入库文件路径集合"""
    with open(RECORD_PATH, "wb") as f:
        pickle.dump(file_set, f)

def load_process_record() -> set:
    """读取持久化的已入库文件记录"""
    if not RECORD_PATH.exists():
        return set()
    with open(RECORD_PATH, "rb") as f:
        return pickle.load(f)

def auto_load_exist_database() -> str:
    db_folder = ROOT_DIR / "config" / "data" / "db"
    chroma_db_file = db_folder / "chroma.sqlite3"
    # 调试打印
    print(f"【项目根目录ROOT_DIR】{ROOT_DIR.resolve()}")
    print(f"【向量库目录】db_folder = {db_folder.resolve()}")
    print(f"【库文件完整路径】chroma_db_file = {chroma_db_file.resolve()}")
    print(f"【文件是否存在】chroma_db_file.exists() = {chroma_db_file.exists()}")

    if not chroma_db_file.exists():
        return "⚠️ 未检测到本地向量库，请点击「完全重构数据库」初始化知识库"

    try:
        # 向量实例存入全局共享字典，多线程可读取
        vec_index = load_exist_vector_index()
        APP_STATE["vec_index"] = vec_index
        APP_STATE["processed_files"] = load_process_record()
        cnt = len(APP_STATE["processed_files"])
        print(f"✅ 向量库实例加载完成，实例对象：{vec_index}")
        return f"✅ 自动加载本地向量库成功\n📄 已加载文档记录数：{cnt}"
    except Exception as e:
        return f"⚠️ 向量库文件存在，但加载失败：{str(e)}\n请关闭程序手动清空缓存后完全重构重建。"

def write_dump_log(doc_list: list):
    with open(DUMP_LOG_PATH, "a", encoding="utf-8") as f:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n==================== 【知识库更新时间：{now}】 ====================\n")
        for idx, doc in enumerate(doc_list):
            meta = doc.metadata
            file_path = meta.get("file_path", "未知路径")
            file_type = meta.get("file_type", "txt/md")
            text_content = doc.text.strip()
            if text_content.startswith("%PDF"):
                warn_text = f"\n⚠️ 异常文档[{idx}] | 类型:{file_type} | 文件:{file_path} 检测到PDF二进制头，解析损坏\n"
                f.write(warn_text)
                print(warn_text)
                continue
            log_header = f"\n----- 文档序号:{idx} | 文件类型:{file_type} | 源文件:{file_path} -----\n"
            f.write(log_header)
            f.write(text_content)
            f.write("\n")
    print(f"\n✅ 本次更新文档已保存至日志文件：{DUMP_LOG_PATH.resolve()}")

def clear_database() -> str:
    db_path = ROOT_DIR / "config" / "data" / "db"
    bm25_path = ROOT_DIR / "config" / "bm25_cache.pkl"
    index_cache_path = INDEX_CACHE_DIR
    try:
        # 第一步：释放内存向量实例，释放Chroma sqlite文件锁（解决WinError32占用）
        APP_STATE["vec_index"] = None
        APP_STATE["all_nodes"] = None
        APP_STATE["processed_files"] = set()

        # 先删除无占用风险的缓存文件
        if index_cache_path.exists() and index_cache_path.is_dir():
            shutil.rmtree(index_cache_path)
        if bm25_path.exists():
            os.remove(bm25_path)
        if RECORD_PATH.exists():
            os.remove(RECORD_PATH)

        # 尝试删除向量库文件夹，捕获Windows文件占用异常
        if db_path.exists() and db_path.is_dir():
            try:
                shutil.rmtree(db_path)
                return "✅ 数据库已全部清除\n向量库、元索引缓存、文件记录、缓存全部重置"
            except OSError as e:
                if "WinError 32" in str(e):
                    return "⚠️ 向量库sqlite文件被进程占用！\n内存状态、索引缓存、文件记录已清空。\n操作方案：关闭程序，手动删除文件夹 config/data/db 后重启执行完全重构。"
                else:
                    raise e
        return "✅ 内存状态、索引缓存、记录文件已清空；向量库文件夹不存在。"
    except Exception as e:
        return f"❌ 清除数据库失败：{str(e)}"

def rebuild_full_database() -> str:
    try:
        clear_result = clear_database()
        print(f"清除操作返回提示：{clear_result}")
        # 无论磁盘向量库是否删除成功，直接继续构建，Chroma自动覆盖重建
        print("🔄 开始完全重构数据库...")
        documents = load_all_documents_full()
        if not documents:
            return f"⚠️ {RAW_TXT_PATH} 目录下无有效txt/md/pdf文档，重构失败"
        write_dump_log(documents)
        print(f"✅ 读取有效文档总数：{len(documents)}")
        nodes = split_docs_into_nodes(documents, semantic_mode=False)
        print(f"✅ 文本切片完成，生成节点：{len(nodes)} 个")
        vec_index = build_vector_index(nodes)
        print("✅ 向量索引构建+持久化完成")
        file_set = {doc.metadata.get("file_path", "") for doc in documents}
        save_process_record(file_set)
        # 更新全局状态字典
        APP_STATE["vec_index"] = vec_index
        APP_STATE["all_nodes"] = nodes
        APP_STATE["processed_files"] = file_set
        success_msg = f"""✅ 完全重构数据库完成
📄 加载有效文档：{len(documents)} 个
📝 生成文本节点：{len(nodes)} 个
💾 向量索引+文件记录已持久化存储"""
        # 拼接清除操作占用提示
        if "⚠️" in clear_result:
            success_msg += f"\n{clear_result}"
        return success_msg
    except Exception as e:
        return f"❌ 完全重构失败：{str(e)}"

def rebuild_incremental_database() -> str:
    try:
        vec_index = APP_STATE["vec_index"]
        if vec_index is None:
            return "⚠️ 请先点击「完全重构数据库」初始化基础索引，再执行增量更新"
        print("🔄 开始增量扫描新增文件...")
        all_documents = load_all_documents_full()
        if not all_documents:
            return "⚠️ 目录下无任何有效文档"
        new_documents = []
        old_file_set = APP_STATE["processed_files"]
        for doc in all_documents:
            file_path = doc.metadata.get("file_path", "")
            if file_path and file_path not in old_file_set:
                new_documents.append(doc)
        if not new_documents:
            return "✅ 扫描完成，无新增文档，无需更新"
        new_nodes = split_docs_into_nodes(new_documents, semantic_mode=False)
        if APP_STATE["all_nodes"] is None:
            APP_STATE["all_nodes"] = new_nodes
        else:
            APP_STATE["all_nodes"].extend(new_nodes)
        vec_index.insert_nodes(new_nodes)
        bm25_path = ROOT_DIR / "config" / "bm25_cache.pkl"
        if bm25_path.exists():
            os.remove(bm25_path)
        # 更新已入库文件集合
        for doc in new_documents:
            file_path = doc.metadata.get("file_path", "")
            if file_path:
                APP_STATE["processed_files"].add(file_path)
        save_process_record(APP_STATE["processed_files"])
        write_dump_log(new_documents)
        return f"""✅ 增量重构完成
📄 新增文档：{len(new_documents)} 个
📝 新增文本节点：{len(new_nodes)} 个
💾 已追加到现有向量索引，文件记录已更新"""
    except Exception as e:
        return f"❌ 增量重构失败：{str(e)}"

def chat_stream(user_msg: str, chat_history: list):
    if chat_history is None:
        chat_history = []
    chat_history.append({"role": "user", "content": user_msg})
    yield chat_history, "等待知识库检索..."
    chat_history.append({"role": "assistant", "content": "思考中..."})
    yield chat_history, "等待知识库检索..."

    # 从全局共享字典读取向量实例，多线程稳定有效
    vec_index = APP_STATE["vec_index"]
    if not vec_index:
        ans_text = "系统异常：知识库未初始化成功，请点击「完全重构数据库」加载知识库"
        ref_content = ""
    elif not user_msg.strip():
        ans_text = "请输入有效的问题"
        ref_content = ""
    else:
        rag_res = rag_full_workflow(user_msg, vec_index)
        if isinstance(rag_res, dict):
            ans_text = rag_res.get("answer", "无匹配回答")
            node_list = rag_res.get("retrieve_nodes", [])
            ref_blocks = []
            for idx, node_item in enumerate(node_list):
                node = node_item.node
                full_context = node.metadata.get("window", node.text).strip()
                source_path = node.metadata.get("file_path", "未知文档")
                page_num = node.metadata.get("page", "")
                page_tip = f"第{page_num}页" if page_num else ""
                if len(full_context) > 1600:
                    full_context = full_context[:1600] + "\n……内容过长已截断"
                block_header = f"【片段{idx+1}】来源：{source_path} {page_tip}\n"
                ref_blocks.append(block_header + full_context)
            ref_content = "\n\n————————————————\n\n".join(ref_blocks)
        else:
            ans_text = str(rag_res)
            ref_content = ""
    chat_history[-1]["content"] = ans_text
    yield chat_history, ref_content

def launch_webui():
    theme = gr.themes.Glass(
        primary_hue="blue",
        secondary_hue="violet",
        neutral_hue="slate",
        radius_size="lg",
    )
    # 程序启动自动加载存量向量库，获取初始状态文本
    init_status_msg = auto_load_exist_database()
    with gr.Blocks(title="本地文档智能问答系统") as demo:
        # 标题卡片
        with gr.Group(elem_classes="module-card"):
            gr.Markdown("# 📚 本地文档智能问答", elem_classes="main-title")
            gr.Markdown("Pymupdf专业PDF解析 · BM25+向量混合检索 · 纯本地私有化部署", elem_classes="sub-title")
        # 数据库管理
        with gr.Group(elem_classes="module-card"):
            gr.Markdown("### 🗄️ 数据库管理")
            with gr.Row():
                clear_btn = gr.Button("🗑️ 清除数据库", variant="stop", scale=1)
                full_rebuild_btn = gr.Button("🔄 完全重构数据库", variant="primary", scale=1)
                inc_rebuild_btn = gr.Button("➕ 动态增量更新", variant="secondary", scale=1)
            status_text = gr.Textbox(
                label="系统状态",
                value=init_status_msg,
                interactive=False,
                lines=4,
                elem_classes="status-box"
            )
        # 聊天问答完整功能区
        with gr.Group(elem_classes="module-card"):
            gr.Markdown("### 💬 智能问答")
            chatbot = gr.Chatbot(height=540)
            with gr.Row():
                msg_input = gr.Textbox(
                    label="",
                    placeholder="输入你的问题，按回车或点击发送...",
                    lines=2,
                    scale=9
                )
                send_btn = gr.Button("发送", variant="primary", scale=1, min_width=80)
            with gr.Accordion("🔍 检索召回原文片段", open=False):
                ref_textbox = gr.Textbox(label="", interactive=False, lines=11)
            with gr.Row():
                clear_chat_btn = gr.Button("清空对话记录", variant="secondary", size="sm")
        # 事件绑定
        clear_btn.click(fn=clear_database, outputs=status_text)
        full_rebuild_btn.click(fn=rebuild_full_database, outputs=status_text)
        inc_rebuild_btn.click(fn=rebuild_incremental_database, outputs=status_text)
        send_btn.click(
            fn=chat_stream,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, ref_textbox]
        ).then(lambda: "", outputs=[msg_input])
        msg_input.submit(
            fn=chat_stream,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, ref_textbox]
        ).then(lambda: "", outputs=[msg_input])
        clear_chat_btn.click(lambda: [], None, outputs=chatbot, queue=False)
    demo.queue()
    demo.launch(
        theme=theme,
        css=CUSTOM_CSS,
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        prevent_thread_lock=False
    )

if __name__ == "__main__":
    launch_webui()
