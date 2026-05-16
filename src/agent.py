from langchain_ollama import ChatOllama
from langsmith import traceable
from langgraph.prebuilt import create_react_agent
from src.tools import search_document
from src.tracer import get_logger

logger = get_logger("agent")

MODEL = "llama3.2:3b"
_llm = ChatOllama(model=MODEL)
_agent = create_react_agent(_llm, tools=[search_document])


@traceable
def run_agent(question: str) -> str:
    logger.info(f"=== Agent query: {question} ===")
    result = _agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content
