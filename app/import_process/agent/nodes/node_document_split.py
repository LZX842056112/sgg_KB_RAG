import json
import os
import re
from pathlib import Path
from typing import Tuple, List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task

# ====================== 全局配置（可根据模型调整）======================
CHUNK_MAX_SIZE = 500  # 500字符串 触发二次切割!
# 单个文本块最大长度（控制不超过模型上下文）
CHUNK_SIZE = 200  # 小值方便测试切割
# 块之间重叠长度（保证语义不丢失）
CHUNK_OVERLAP = 20


@step_log("step_1_validate_clean")
def step_1_validate_clean(state) -> Tuple[str, str]:
    """
    1. 获取数据  md_content , file_title , md_path
    2. md_content进行非空校验 空->异常
    3. md_content不为空 -> 数据清洗 -> 字符串统一替换
    4. file_title进行非空判断 -> 空 -> 通过md_path获取file_title
    5. 返回数据即可
    :param state:
    :return:
    """
    md_content = state['md_content']
    file_title = state['file_title']
    md_path = state['md_path']

    # 进行必要非空校验
    if not md_content:
        logger.warning(f"没有从state读取到md_content内容,我们使用md_path尝试再次读取!")
        if md_path:
            md_content = Path(md_path).read_text(encoding="utf-8")
            state['md_content'] = md_content
        if not md_content:
            logger.error(f"md_content没数据,并且尝试读取md_path依然没有数据,终止执行!!")
            raise ValueError("md_content没数据,并且尝试读取md_path依然没有数据,终止执行!!")
    if not file_title:
        logger.warning("没有在state读取到file_title,计划给与默认值!")
        if md_path:
            file_title = Path(md_path).stem
        if not file_title:
            file_title = "default"
        state['file_title'] = file_title

    # md_content内容进行清洗
    """
      \r\n
      \r     -> \n
    """
    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
    # 返回结果
    return md_content, file_title


@step_log("step_2_data_chunk_title")
def step_2_data_chunk_title(md_content, file_title) -> List[Dict[str, str]]:
    """
      1. 定义正则
      2. 根据\n进行行的切割
      3. 准备一些数据容器 chunks = [] 整体结果  current_title = str  current_title_lines = []  is_code_block = False  chunk_size = 0
      4. 循环处理每行数据
      5. 检查是否进入和出去哪一行是否是代码块
      6. 判断是不是标题 以及 不是代码块  -> 标题处理
      7. 不是代码块 也不是标题 -> 普通行处理
      8. 跳出循环后处理最后一个标题的内容
      9. 检查没有标题的文档 -> title default  md_content
      10. 返回结果
    :param md_content:
    :param file_title:
    :return:
    """
    # 1. 定义正则
    reg = re.compile(r"^\s*#{1,6}\s.+")
    # 2. 根据\n进行行的切割
    lines = md_content.split("\n")
    # 3. 准备一些数据容器 chunks = [] 整体结果  current_title = str  current_title_lines = []  is_code_block = False  chunk_size = 0
    chunks = []  # 最终结果!
    current_title = None  # 记录当前标题
    current_title_lines = []  # 记录当前标题的行内容
    is_code_block = False  # 是否进入代码块,默认没有
    chunk_size = 0  # 切的块数
    # 4. 循环处理每行数据
    for line in lines:
        # 5. 检查是否进入和出去哪一行是否是代码块
        if line.startswith("```") or line.startswith("~~~"):
            # ```  ```
            is_code_block = not is_code_block
            current_title_lines.append(line)
            continue

        # 6. 判断是不是标题 以及 不是代码块  -> 标题处理
        if reg.match(line) and not is_code_block:
            # 满足标题的结果 # xxx 并且不在代码块!
            # 真标题
            # 结算上一个标题的内容
            if current_title:
                # 第二个之后的标题
                # 将上一份标题内容存储chunks内部
                chunks.append({
                    # current_title_lines = [行,行,行,行]
                    "content": "\n".join(current_title_lines),
                    "title": current_title,
                    "file_title": file_title
                })
            # 清空旧数据
            current_title = line  # 本次行,赋予标题
            current_title_lines = [current_title]  # 标题作为内容的第一行
            chunk_size += 1
        else:
            # 不是标题! [在代码块 # ]
            # 普通行
            current_title_lines.append(line)

    # 8. 跳出循环后处理最后一个标题的内容
    if current_title:
        chunks.append({
            # current_title_lines = [行,行,行,行]
            "content": "\n".join(current_title_lines),
            "title": current_title,
            "file_title": file_title
        })
    # 9. 检查没有标题的文档 -> title default  md_content
    if chunk_size == 0:
        chunks.append({
            # current_title_lines = [行,行,行,行]
            "content": md_content,
            "title": "default",
            "file_title": file_title
        })
        chunk_size = 1
    # 10. 返回结果
    logger.info(f"完成语义切割,切块数量:{chunk_size},内容:{chunks[:3]}")
    return chunks


@step_log("step_3_data_refine_chunk")
def step_3_data_refine_chunk(chunks) -> List[Dict[str, str]]:
    """
    作用: 将超过执行size的标题内容,1000字符,进行二次切分,二次切分产生:  parent_title part
    入参: chunks
    出参: chunks   part parent_title
    步骤:
      1. 先定义langchain提供的递归切割器 [块大小,重叠部分,切割符号]
      2. 循环原来的chunks
      3. 获取每块的content长度
      4. 没有超过 -> 不需要 -> part parent_title
      4. 超过 -> 递归切割器 二次切割
      5. 最终记录结果 final_chunks
      6. 返回结果即可
    """
    # . 先定义langchain提供的递归切割器 [块大小,重叠部分,切割符号]
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "；", " "],  # 字符级切分，最容易观察 overlap
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    # 2. 获取原来的chunks
    final_chunks = []
    for chunk in chunks:
        # 3.判断chunk content内容长度
        # chunk -> 父块
        content = chunk['content']
        if len(content) <= CHUNK_MAX_SIZE:
            # 不需要切
            # 补齐 parent_title part
            chunk["part"] = 1
            chunk["parent_title"] = chunk["title"]
            final_chunks.append(chunk)
        else:
            # 切后的文本块
            # 小块
            spliter_chunks = splitter.split_text(content)
            for index, text in enumerate(spliter_chunks, start=1):
                final_chunks.append({
                    "content": text,
                    "title": f'{chunk["title"]}_{index}',  # title_1  _2  _3
                    "file_title": chunk["file_title"],
                    "part": index,
                    "parent_title": chunk["title"]
                })
    return final_chunks


@step_log("step_5_backup_data")
def step_5_backup_data(chunks, md_path):
    """
    将数据存储到本地 文件夹名 / chunks.json
    """
    chunks_json_path_obj = Path(md_path).parent / "chunks.json"
    chunks_json_path_obj.write_text(json.dumps(chunks, ensure_ascii=False, indent=4), encoding="utf-8")


@node_log("node_document_split")
def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    为什么叫这个名字: 将长文档切分成小的 Chunks (切片) 以便检索。
    未来要实现:
    """
    # 1. 任务 + 日志处理
    add_running_task(state['task_id'], "node_document_split")
    # 2. 数据校验和清晰 step_1_validate_clean
    md_content, file_title = step_1_validate_clean(state)
    # 3. 按照语义进行数据切割 step_2_data_chunk_title [文档中没有标题]
    # [{content , title , file_title}]
    chunks = step_2_data_chunk_title(md_content, file_title)
    # 4. 检查有没有超过指定的块,进行二次细分切割 step_3_data_refine_chunk
    chunks = step_3_data_refine_chunk(chunks)
    # 5. 完善数据格式,补齐part parent_title step_4_data_deal
    # 6. 数据备份处理,chunk.json  step_5_backup_data
    step_5_backup_data(chunks, state['md_path'])
    # 7. 修改state属性 chunks -> chunk切块列表即可
    state["chunks"] = chunks
    add_done_task(state['task_id'], "node_document_split")
    return state


if __name__ == '__main__':
    """
    单元测试：联合node_md_img（图片处理节点）进行集成测试
    测试条件：1.已配置.env（MinIO/大模型环境） 2.存在测试MD文件 3.能导入node_md_img
    测试流程：先运行图片处理→再运行文档切分，验证端到端流程
    """

    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    from app.import_process.agent.nodes.node_md_img import node_md_img

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\hak180使用说明书", "hak180使用说明书.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": "",
            "file_title": "hak180使用说明书",
            "local_dir": os.path.join(PROJECT_ROOT, "output"),
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
        logger.info("\n=== 开始执行文档切分节点集成测试 ===")

        logger.info(">> 开始运行当前节点：node_document_split（文档切分）")
        final_state = node_document_split(result_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"✅ 测试成功：最终生成{len(final_chunks)}个有效Chunk{final_chunks}")
