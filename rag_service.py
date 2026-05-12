import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from qdrant_client.models import FieldCondition, Filter, MatchValue

from main import COLLECTION_NAME, build_vector_store


load_dotenv(dotenv_path=Path(".env"))


class RAGState(TypedDict, total=False):
    question: str
    user_id: str
    search_query: str
    documents: list[Document]
    context: str
    has_context: bool
    answer: str


def configure_langsmith() -> None:
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        return
    project = os.getenv("LANGSMITH_PROJECT", "rag-qdrant-src")
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", project)
    os.environ.setdefault("LANGCHAIN_PROJECT", project)
    os.environ.setdefault("LANGCHAIN_API_KEY", api_key)


def format_docs(docs: list[Document]) -> str:
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Doc {i} | {source} | trang {page}]\n{doc.page_content}")
    return "\n\n".join(formatted)


def missing_context_answer() -> str:
    return "Mình chưa tìm thấy thông tin đủ rõ trong tài liệu để trả lời câu hỏi này."


configure_langsmith()

top_k = int(os.getenv("RAG_TOP_K", "10"))
vector_store = build_vector_store()

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
)

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", """Bạn tối ưu câu hỏi tiếng Việt để tìm kiếm trong kho tài liệu.
Yêu cầu:
- Giữ nguyên ý định của người hỏi.
- Viết lại thành một câu hỏi ngắn, rõ, giàu từ khóa.
- Không trả lời câu hỏi.
- Chỉ xuất ra câu hỏi đã viết lại."""),
    ("human", "{question}"),
])
rewrite_chain = rewrite_prompt | llm | StrOutputParser()

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", """Bạn kiểm tra CONTEXT có đủ thông tin để trả lời QUESTION không.
Trả lời chính xác một từ:
- yes: nếu CONTEXT có thông tin liên quan và đủ để trả lời.
- no: nếu CONTEXT không liên quan hoặc không đủ thông tin."""),
    ("human", "QUESTION:\n{question}\n\nCONTEXT:\n{context}"),
])
grade_chain = grade_prompt | llm | StrOutputParser()

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """Bạn là trợ lý trả lời câu hỏi dựa trên tài liệu được cung cấp.
Quy tắc:
- Chỉ dùng thông tin trong CONTEXT bên dưới để trả lời.
- Trả lời bằng tiếng Việt, ngắn gọn, chính xác.
- Khi trích thông tin, có thể nêu nguồn gồm tên file và số trang nếu có trong metadata.
- Nếu CONTEXT không đủ thông tin, hãy nói rằng bạn chưa tìm thấy thông tin trong tài liệu.

CONTEXT:
{context}"""),
    ("human", "{question}"),
])
answer_chain = answer_prompt | llm | StrOutputParser()


def rewrite_question(state: RAGState) -> RAGState:
    question = state["question"].strip()
    rewritten = rewrite_chain.invoke({"question": question}).strip()
    return {"search_query": rewritten or question}


def retrieve_documents(state: RAGState) -> RAGState:
    query = state.get("search_query") or state["question"]
    user_id = state.get("user_id", "")

    filter_cond = (
        Filter(must=[FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id))])
        if user_id else None
    )

    docs = vector_store.similarity_search(query, k=top_k, filter=filter_cond)
    return {"documents": docs, "context": format_docs(docs)}


def grade_context(state: RAGState) -> RAGState:
    documents = state.get("documents", [])
    context = state.get("context", "")
    if not documents or not context.strip():
        return {"has_context": False}
    grade = grade_chain.invoke({"question": state["question"], "context": context})
    return {"has_context": grade.strip().lower().startswith("yes")}


def generate_answer(state: RAGState) -> RAGState:
    answer = answer_chain.invoke({
        "question": state["question"],
        "context": state.get("context", ""),
    })
    return {"answer": answer}


def answer_not_found(_: RAGState) -> RAGState:
    return {"answer": missing_context_answer()}


def route_after_grading(state: RAGState) -> str:
    return "generate_answer" if state.get("has_context") else "answer_not_found"


workflow = StateGraph(RAGState)
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("grade_context", grade_context)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("answer_not_found", answer_not_found)
workflow.add_edge(START, "rewrite_question")
workflow.add_edge("rewrite_question", "retrieve_documents")
workflow.add_edge("retrieve_documents", "grade_context")
workflow.add_conditional_edges("grade_context", route_after_grading, {
    "generate_answer": "generate_answer",
    "answer_not_found": "answer_not_found",
})
workflow.add_edge("generate_answer", END)
workflow.add_edge("answer_not_found", END)

rag_graph = workflow.compile()


def run_rag(question: str, user_id: str = "") -> str:
    state = rag_graph.invoke({"question": question, "user_id": user_id})
    return state.get("answer", missing_context_answer())
