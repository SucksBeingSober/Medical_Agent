import os
import re
from pathlib import Path
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ==================== 0. 环境配置 ====================
load_dotenv()
# 必须在加载 HuggingFace 相关模型之前设置镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ==================== 1. 依赖导入 ====================
import pymupdf
import pymupdf4llm

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_tavily import TavilySearch
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Reranker 相关
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# LangGraph 相关
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

# ==================== 2. 基础参数 ====================
# 请替换为你的实际 PDF 路径
PDF_PATH = Path(r"D:\HuaweiMoveData\Users\64346\Desktop\临床常见病诊疗指南.pdf")

EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RETRIEVAL_K = 15
RERANKER_TOP_N = 3
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# ==================== 3. 模型与数据库初始化 ====================
search = TavilySearch(max_results=5)

llm = ChatOpenAI(
    model=os.getenv("SILICONFLOW_MODEL_ID"),
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=os.getenv("SILICONFLOW_BASE_URL"),
    temperature=0
)

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=os.getenv("SILICONFLOW_BASE_URL"),
    check_embedding_ctx_length=False
)


# ==================== 4. PDF 解析与分块逻辑（融合版） ====================
def has_markdown_headers(text: str) -> bool:
    pattern = r"(?m)^#{1,3}[ \t]+\S"
    return bool(re.search(pattern, text))


def pdf_to_markdown(pdf_path: Path) -> str:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在：{pdf_path}")

    print("\n========== PDF 转 Markdown ==========")
    with pymupdf.open(str(pdf_path)) as doc:
        print(f"PDF 页数：{doc.page_count}")
        markdown_text = ""

        try:
            pymupdf4llm.use_layout(True)
            print("正在使用 Layout Mode 解析 PDF...")
            markdown_text = pymupdf4llm.to_markdown(doc, show_progress=True)
        except Exception as e:
            print(f"⚠️ Layout Mode 解析失败：{e}")

        if not markdown_text.strip() or not has_markdown_headers(markdown_text):
            print("\n未检测到足够的 Markdown 标题标记，切换兜底模式...")
            try:
                pymupdf4llm.use_layout(False)
                toc = doc.get_toc()
                if toc:
                    print("✅ 检测到 PDF 内部目录，使用 TocHeaders")
                    header_info = pymupdf4llm.TocHeaders(doc)
                else:
                    print("未检测到 PDF 内部目录，根据字体大小推断...")
                    header_info = pymupdf4llm.IdentifyHeaders(doc, max_levels=3)

                fallback_markdown = pymupdf4llm.to_markdown(doc, hdr_info=header_info, show_progress=True)
                if fallback_markdown.strip():
                    if has_markdown_headers(fallback_markdown) or not markdown_text.strip():
                        markdown_text = fallback_markdown
            except Exception as e:
                print(f"⚠️ 标题识别兜底失败：{e}")

        if not markdown_text.strip():
            raise ValueError("PDF 解析结果为空，请检查文件。")
    return markdown_text


def split_documents(markdown_text: str, pdf_path: Path) -> list[Document]:
    print("\n========== 开始智能分块 ==========")
    headers_to_split_on = [("#", "一级标题"), ("##", "二级标题"), ("###", "三级标题")]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=True)
    header_splits = markdown_splitter.split_text(markdown_text)
    header_splits = [doc for doc in header_splits if doc.page_content.strip()]

    if not header_splits:
        header_splits = [Document(page_content=markdown_text, metadata={})]

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""],
        add_start_index=True
    )
    splits = recursive_splitter.split_documents(header_splits)
    splits = [doc for doc in splits if doc.page_content.strip()]

    for index, chunk in enumerate(splits):
        chunk.metadata["source"] = str(pdf_path)
        chunk.metadata["chunk_id"] = index

    print(f"✅ 分块完成，最终文本块数量：{len(splits)}")
    return splits


def format_docs(docs: list[Document]) -> str:
    """把检索到的块拼成上下文，带上章节和来源"""
    formatted = []
    for doc in docs:
        headers = []
        for key in ["一级标题", "二级标题", "三级标题"]:
            if doc.metadata.get(key):
                headers.append(doc.metadata[key])
        header_path = " > ".join(headers) if headers else "未识别章节"
        source = Path(doc.metadata.get("source", "未知来源")).name

        formatted.append(f"【章节：{header_path}；来源：{source}】\n{doc.page_content}")
    return "\n\n".join(formatted)


# ----------------- 执行解析、分块、构建 RAG 检索器 -----------------
print("正在构建医疗 RAG 知识库...")
markdown_text = pdf_to_markdown(PDF_PATH)
# 保存 PDF 解析后的 Markdown，方便人工检查
debug_md_path = PDF_PATH.with_name(
    PDF_PATH.stem + "_解析结果.md"
)

debug_md_path.write_text(
    markdown_text,
    encoding="utf-8"
)
print(f"✅ PDF 解析结果已保存：{debug_md_path}")
splits = split_documents(markdown_text, PDF_PATH)

vectorstore = InMemoryVectorStore.from_documents(documents=splits, embedding=embeddings)

# 粗排检索器
base_retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})

# 初始化 Reranker（精排）
print("\n正在加载 Reranker 模型...")
cross_encoder_model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
reranker = CrossEncoderReranker(model=cross_encoder_model, top_n=RERANKER_TOP_N)

# 融合检索器（粗排 + 精排）
retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever
)
print("✅ 向量数据库与 Reranker 建立成功\n")


# ==================== 5. 工具与 Agent 初始化 ====================
@tool
def search_medical_guideline(query: str):
    """当需要查询权威的常见病诊疗指南、用药规范时优先调用此工具。"""
    docs = retriever.invoke(query)
    return format_docs(docs)


agent = create_agent(
    model=llm,
    tools=[search, search_medical_guideline],
    system_prompt="你是一个工作在医疗平台的专业AI助手。遇到病情诊断时，请优先调用search_medical_guideline查阅本地诊疗指南；如果本地指南未涵盖，再调用tavily_search 联网搜索。"
)


# ==================== 6. LangGraph 状态与节点定义 ====================
class PatientState(TypedDict):
    symptom: str
    name: str
    sex: str
    age: str
    health_insurance: bool | None
    emergency_flag: bool
    info_approved: bool
    diagnosis_room: str
    diagnosis: str
    suggest: str
    more_question: str
    messages: Annotated[list, add_messages]


class TriageResult(BaseModel):
    diagnosis_room: str = Field(
        description="建议就诊的医学科室名称。必须是实际医院科室，例如耳鼻咽喉科、呼吸内科、消化内科、全科等。禁止填写年龄、性别、患者身份或症状。"
    )
    diagnosis: str = Field(
        description="可能病因分析。必须分析症状背后可能的疾病或原因，不能只重复患者的症状。应说明不确定性，不得将可能病因表述为确诊结果。"
    )
    suggest: str = Field(
        description="具体的就医和健康管理建议。应结合患者症状、年龄、性别及检索到的指南，给出有针对性的建议、需要补充的信息以及危险信号。不能只输出客套话或建议患者自行确诊。"
    )

# 直接使用结构化输出，不再先生成普通文本再二次提取
structured_llm = llm.with_structured_output(
    TriageResult,
    method="function_calling"
)


def input_symptom(state: PatientState) -> PatientState:
    print(f"收到症状信息:{state['symptom']}，正在检查个人信息是否完整...")
    return state


def check_info(state: PatientState) -> PatientState:
    if not state["name"] or not state["sex"] or not state["age"] or not state["symptom"] or state[
        "health_insurance"] is None:
        print("信息不完整，请完善信息")
        state["info_approved"] = False
    else:
        print("信息完整，将进行下一步")
        state["info_approved"] = True
    return state


def supplement_info(state: PatientState) -> PatientState:
    if state["name"] == "": state["name"] = input("姓名信息缺失，请补全：")
    if state["symptom"] == "": state["symptom"] = input("症状信息缺失，请补全：")
    if state["sex"] == "": state["sex"] = input("性别信息缺失，请补全：")
    if state["age"] == "": state["age"] = input("年龄信息缺失，请补全：")
    if state["health_insurance"] is None:
        h = input("医保信息缺失，请补全：1.有 2.无\n")
        state["health_insurance"] = (h == "1")
    return state


def screen_emergency(state: PatientState) -> PatientState:
    text = state["symptom"]
    emergency_words = ["胸痛", "呼吸困难", "意识不清", "大出血"]
    if any(word in text for word in emergency_words):
        state["diagnosis"] = "⚠️ 危险信号！请立即前往急诊或拨打 120！不要依赖本系统。"
        state["emergency_flag"] = True
    else:
        state["emergency_flag"] = False
    return state

def generate_diagnosis(state: PatientState) -> PatientState:
    """
    医疗分诊节点：
    1. 根据患者信息构造检索问题
    2. 强制执行本地 RAG 检索
    3. 将检索结果交给结构化 LLM
    4. 直接生成三个结构化字段
    """

    symptom = state["symptom"]
    sex = state["sex"]
    age = state["age"]
    name = state["name"]
    health_insurance = state["health_insurance"]

    print("\n========== 开始医疗分诊 ==========")

    # ==================== 1. 构造 RAG 检索问题 ====================

    retrieval_query = f"""
    患者主要症状：{symptom}

    患者年龄：{age}岁
    患者性别：{sex}

    请重点检索：
    1. 与该症状有关的常见疾病
    2. 相关疾病的临床表现
    3. 推荐就诊科室
    4. 常见危险信号
    5. 相关诊疗建议
    """

    # ==================== 2. 强制执行本地 RAG 检索 ====================

    print("\n正在检索本地医疗指南...")

    try:
        docs = retriever.invoke(retrieval_query)

        print(f"✅ RAG 检索完成，共获取 {len(docs)} 个相关文本块")

        # 输出检索结果，方便检查 RAG 是否真正工作
        for index, doc in enumerate(docs, start=1):
            print(f"\n--- RAG 检索结果 {index} ---")

            headers = []
            for key in ["一级标题", "二级标题", "三级标题"]:
                if doc.metadata.get(key):
                    headers.append(doc.metadata[key])

            print("章节：", " > ".join(headers) if headers else "未识别章节")
            print("内容：", doc.page_content[:500])

        guideline_context = format_docs(docs)

        if not guideline_context.strip():
            guideline_context = "本次本地医疗指南没有检索到相关内容。"

    except Exception as e:
        print(f"⚠️ RAG 检索失败：{e}")
        docs = []
        guideline_context = "本次本地医疗指南检索失败，不能声称已经查阅相关指南。"

    # 限制传给模型的上下文长度
    guideline_context = guideline_context[:12000]

    # ==================== 3. 构造结构化分诊提示词 ====================

    system_prompt = """
你是一名医疗平台的 AI 分诊助手。

你的任务是根据患者信息和提供的医疗指南内容，生成结构化的分诊报告。

必须遵守以下规则：

【规则一：科室判断】
1. diagnosis_room 必须填写真实存在的医学科室。
2. 例如：耳鼻咽喉科、呼吸内科、消化内科、神经内科、全科、急诊科。
3. 绝对不能将患者年龄、性别、姓名或症状填写为科室。
4. 信息不足时，应选择相对合理的初诊科室，并说明判断的不确定性。

【规则二：病因分析】
1. diagnosis 必须分析症状背后可能的病因。
2. 不允许简单重复患者提供的症状。
3. 应根据患者信息和医疗指南分析可能性。
4. 不能仅凭单一症状作出确定诊断。
5. 对于无法确定的情况，必须明确说明需要补充哪些信息。

【规则三：医生建议】
1. suggest 必须给出具体、有针对性的建议。
2. 不允许只输出客套话。
3. 结合患者症状说明建议就诊的科室。
4. 说明需要进一步了解的症状。
5. 必须注意可能存在的危险信号。
6. 不得编造医疗指南中没有出现的内容，也不得随意给出药物剂量。

【规则四：医疗指南】
1. 优先依据提供的本地医疗指南。
2. 如果本地指南没有相关内容，必须明确说明。
3. 可以提供一般医学知识作为辅助判断，但不能冒充指南内容。

【规则五：输出】
严格按照提供的结构化字段输出。
不要在字段中填写与字段含义无关的内容。
"""

    user_prompt = f"""
请根据以下信息完成医疗分诊。

========== 患者信息 ==========

姓名：{name}
性别：{sex}
年龄：{age}岁
主要症状：{symptom}
是否有医保：{"有" if health_insurance else "无"}

========== 本地医疗指南检索结果 ==========

{guideline_context}

========== 分诊任务 ==========

请完成以下工作：

1. 判断患者应该优先就诊的科室。
2. 分析症状可能对应的疾病或原因。
3. 给出针对性的就医建议。
4. 如果目前信息不足，请明确说明还需要了解什么。
5. 说明需要警惕的危险信号。

请特别注意：
患者提供的症状不等于疾病名称；
患者的年龄和性别不等于科室名称；
不要重复患者的信息来凑答案。

请生成完整、准确、谨慎的结构化分诊结果。
"""

    # ==================== 4. 直接生成结构化结果 ====================

    print("\n正在生成结构化分诊报告...")

    try:
        result_obj = structured_llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

        # 检查结构化结果
        print("\n========== 模型结构化输出 ==========")
        print(result_obj.model_dump_json(indent=2))

        # ==================== 5. 结果校验 ====================

        invalid_departments = [
            "男性", "女性", "男", "女",
            "岁", "患者", "喉咙痛"
        ]

        if (
            not result_obj.diagnosis_room.strip()
            or any(word in result_obj.diagnosis_room for word in invalid_departments)
        ):
            raise ValueError(
                f"科室字段不符合要求：{result_obj.diagnosis_room}"
            )

        if (
            not result_obj.diagnosis.strip()
            or result_obj.diagnosis.strip() == symptom.strip()
        ):
            raise ValueError(
                f"病因字段不符合要求：{result_obj.diagnosis}"
            )

        if not result_obj.suggest.strip():
            raise ValueError("医生建议字段为空")

        # ==================== 6. 写入 LangGraph State ====================

        state["diagnosis_room"] = result_obj.diagnosis_room
        state["diagnosis"] = result_obj.diagnosis
        state["suggest"] = result_obj.suggest

        print("\n✅ 结构化分诊结果生成成功")

    except Exception as e:

        print(f"\n❌ 结构化分诊失败：{e}")

        # 不再使用容易造成错误报告的默认值
        state["diagnosis_room"] = "需要人工分诊"
        state["diagnosis"] = (
            "当前无法可靠生成病因分析，不能根据现有信息确定具体疾病。"
        )
        state["suggest"] = (
            "建议由正规医疗机构的医生进一步评估。"
            "请补充症状持续时间、严重程度及伴随症状，"
            "再进行进一步分诊。"
        )

    # ==================== 7. 更新对话历史 ====================

    clean_summary = (
        f"初步分诊建议：建议就诊科室为"
        f"【{state['diagnosis_room']}】。"
        f"可能病因：{state['diagnosis']}。"
    )

    state["messages"].append({
        "role": "assistant",
        "content": clean_summary
    })

    return state


def nurse_review(state: PatientState) -> PatientState:
    decision = interrupt({
        "draft": state["diagnosis_room"],
        "message": "请护士审核科室信息。输入 'approve' 通过，或输入修改意见："
    })
    if decision == "approve":
        state["messages"].append({"role": "system", "content": "护士已审核通过。"})
    else:
        state["diagnosis_room"] = decision
        state["messages"].append({"role": "system", "content": f"护士修改了科室信息：{decision}"})
    return state


# ==================== 7. 构建 LangGraph ====================
workflow = StateGraph(PatientState)
workflow.add_node("input_symptom", input_symptom)
workflow.add_node("check_info", check_info)
workflow.add_node("screen_emergency", screen_emergency)
workflow.add_node("nurse_review", nurse_review)
workflow.add_node("generate_diagnosis", generate_diagnosis)
workflow.add_node("supplement_info", supplement_info)

workflow.set_entry_point("input_symptom")
workflow.add_edge("input_symptom", "check_info")


def info_router(state: PatientState):
    return "screen_emergency" if state.get("info_approved", False) else "supplement_info"


workflow.add_conditional_edges(
    source="check_info",
    path=info_router,
    path_map={"screen_emergency": "screen_emergency", "supplement_info": "supplement_info"}
)
workflow.add_edge("supplement_info", "check_info")


def emergency_router(state: PatientState):
    return END if state.get("emergency_flag", False) else "generate_diagnosis"


workflow.add_conditional_edges(
    source="screen_emergency",
    path=emergency_router,
    path_map={"generate_diagnosis": "generate_diagnosis", END: END}
)
workflow.add_edge("generate_diagnosis", "nurse_review")
workflow.add_edge("nurse_review", END)

memory = InMemorySaver()
app = workflow.compile(checkpointer=memory)

# ==================== 8. 运行入口 ====================
if __name__ == '__main__':
    symptom = input("请输入您的症状：")
    name = input("请输入您的姓名：")
    sex = input("请输入您的性别：")
    age = input("请输入您的年龄：")
    health_insurance = input("您是否有医保呢？1.有 2.无\n")

    initial_state = {
        "symptom": symptom, "name": name, "sex": sex, "age": age, "health_insurance": health_insurance == "1",
        "info_approved": False, "diagnosis": "", "diagnosis_room": "", "suggest": "", "more_question": "",
        "emergency_flag": False,
        "messages": [
            {"role": "user", "content": f"我叫{name},{sex},{age}岁。我的症状是:{symptom}。医保情况:{health_insurance}"}]
    }

    print("\n--- 开始病患咨询 ---")
    config = {"configurable": {"thread_id": "patient_001"}}
    final_state = app.invoke(initial_state, config=config)

    if final_state.get("emergency_flag"):
        print("\n" + "=" * 30 + "\n🚨 紧急医疗警报！\n" + "=" * 30)
        print(final_state["diagnosis"])
        print("⚠️ 免责声明：本报告仅供健康咨询参考...")
        exit(0)

    current_state = app.get_state(config)
    if current_state.next:
        print(f"\n[系统暂停] 护士请审核科室信息：{current_state.values['diagnosis_room']}")
        doc_input = input("请输入审核结果（输入 'approve' 通过，或输入修改意见）：")
        final_state = app.invoke(Command(resume=doc_input), config=config)

    print("\n" + "=" * 30 + "\n🏥 问诊报告单\n" + "=" * 30)
    print(f"👤 患者: {final_state['name']} | {final_state['sex']} | {final_state['age']}岁")
    print(f"🩺 症状: {final_state['symptom']}")
    print("=" * 30)
    print(f"🏥 分诊科室: {final_state['diagnosis_room']}")
    print(f"🔍 可能病因: {final_state['diagnosis']}")
    print(f"💡 医生建议:\n {final_state['suggest']}")
    print("=" * 30)

    print("\n--- 进入自由答疑模式 ---")
    while True:
        q = input("您还有其他问题吗？（输入'exit'结束）\n")
        if "exit" in q:
            break

        final_state["messages"].append({"role": "user", "content": q})
        chat_messages = [
                            {"role": "system",
                             "content": "你现在处于自由答疑模式。请直接、自然、口语化地回答患者的问题。绝对不要输出任何 JSON 代码，也不允许使用Markdown格式!!"}
                        ] + final_state["messages"]

        result = agent.invoke({"messages": chat_messages})
        content_text = result["messages"][-1].content

        # 强制清洗可能残留的 Markdown
        content_text = re.sub(r'#+\s*', '', content_text)
        content_text = re.sub(r'\*\*(.*?)\*\*', r'\1', content_text)
        content_text = re.sub(r'\*(.*?)\*', r'\1', content_text)
        content_text = re.sub(r'^\s*[-*]\s+', '• ', content_text, flags=re.MULTILINE)

        print(f"\nAI医生: {content_text}\n")
        final_state["messages"].append({"role": "assistant", "content": content_text})

    print("\n⚠️ 免责声明：本报告仅供健康咨询参考，不能替代专业医师的诊断，请及时前往正规医院就诊。")