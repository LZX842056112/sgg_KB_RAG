# 加载环境变量：从 .env 文件读取配置（如Milvus地址、KG服务地址、BGE模型路径等）
from dotenv import load_dotenv
# 导入LangGraph核心依赖：StateGraph(状态图)、START/END(内置起始/结束节点常量)
from langgraph.graph import StateGraph, END, START

from app.core.logger import logger
# 导入自定义状态类：统一管理工作流全程的所有数据（各节点共享/修改）
from app.import_process.agent.state import ImportGraphState, create_default_state
# 导入所有自定义业务节点：每个节点对应知识库导入的一个具体步骤
from app.import_process.agent.nodes.node_entry import node_entry  # 入口节点：初始化参数、校验输入
from app.import_process.agent.nodes.node_pdf_to_md import node_pdf_to_md  # PDF转MD：解析PDF文件为markdown格式
from app.import_process.agent.nodes.node_md_img import node_md_img  # MD图片处理：提取/下载markdown中的图片、修复图片路径
from app.import_process.agent.nodes.node_document_split import node_document_split  # 文档分块：将长文档切分为符合模型要求的小片段
from app.import_process.agent.nodes.node_item_name_recognition import node_item_name_recognition  # 项目名识别：从分块中提取核心项目名称（业务定制化）
from app.import_process.agent.nodes.node_bge_embedding import node_bge_embedding  # BGE向量化：将文本分块转换为向量表示（适配Milvus向量库）
from app.import_process.agent.nodes.node_import_milvus import node_import_milvus  # 导入Milvus：将向量数据写入Milvus向量数据库


# 初始化环境变量：必须在配置读取前执行，确保后续节点能获取到环境变量中的配置信息
load_dotenv()

# 1. 定义状态图对象,并且指定全局state类型
workflow = StateGraph(ImportGraphState)
# 2. 添加节点
workflow.add_node("node_entry",node_entry)
workflow.add_node("node_pdf_to_md",node_pdf_to_md)
workflow.add_node("node_md_img",node_md_img)
workflow.add_node("node_document_split",node_document_split)
workflow.add_node("node_item_name_recognition",node_item_name_recognition)
workflow.add_node("node_bge_embedding",node_bge_embedding)
workflow.add_node("node_import_milvus",node_import_milvus)

# 3. 指定入口节点
# workflow.add_edge(START,"node_entry")
workflow.set_entry_point("node_entry")
# 4. 设置入口节点后的条件边
# node_entry 的后面,判断文件类型,转发到对应的节点
# node_entry -> state -> is_md_read_enabled = True  or  is_pdf_read_enabled = True  or 都是False
# is_md_read_enabled = True -> node_md_img
# is_pdf_read_enabled = True -> node_pdf_to_md
# 都是False -> END
def after_entry_node(state: ImportGraphState):
    if state['is_md_read_enabled']:
        return "node_md_img"
    elif state['is_pdf_read_enabled']:
        return "node_pdf_to_md"
    else:
        return END
"""
添加条件边
  参数1: 原节点
  参数2: 路由函数
  参数3: path_map [可选] 推荐
        什么时候可以省略: 路由函数返回的字符串刚好等于目标节点名称 可以省略 path_map 
        什么时候不能省略: 1. 路由函数返回的字符串不等于节点名称的时候 2. 如果你想要显示的打印图的结构必须显示添加
"""
workflow.add_conditional_edges("node_entry", after_entry_node,{
    "node_md_img": "node_md_img",
    "node_pdf_to_md": "node_pdf_to_md",
    END: END
})
# 5. 设置静态条件边
workflow.add_edge("node_pdf_to_md", "node_md_img")
workflow.add_edge("node_md_img", "node_document_split")
workflow.add_edge("node_document_split", "node_item_name_recognition")
workflow.add_edge("node_item_name_recognition", "node_bge_embedding")
workflow.add_edge("node_bge_embedding", "node_import_milvus")
workflow.add_edge("node_import_milvus", END)

# 6. 编译图对象即可
kb_import_app = workflow.compile()











