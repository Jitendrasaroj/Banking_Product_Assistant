from BankingAssistant_agent.agent.state import AgentState
from BankingAssistant_agent.config import settings
from BankingAssistant_agent.guardrails.nlu_guard import check_input_nlu
from BankingAssistant_agent.guardrails.output_guard import check_output
from BankingAssistant_agent.guardrails.regex_guard import check_input_regex
from BankingAssistant_agent.rag.context_builder import build_context
from BankingAssistant_agent.rag.generator import generate_answer
from BankingAssistant_agent.rag.query_rewriter import rewrite_query
from BankingAssistant_agent.rag.retriever import retrieve

REFUSALS = {
    "regex": "Sorry, I can't process that message. Please remove personal details or instructions to the assistant.",
    "nlu": "Sorry, I can only help with banking-related questions.",
    "output": "Sorry, I couldn't find a reliable answer to that in my banking documents.",
}


def regex_guard_node(state: AgentState) -> dict:
    return {"guardrail_results": [check_input_regex(state["question"])]}


def nlu_guard_node(state: AgentState) -> dict:
    return {"guardrail_results": [check_input_nlu(state["question"])]}


def rewrite_query_node(state: AgentState) -> dict:
    return {"search_query": rewrite_query(state["question"])}


def retrieve_node(state: AgentState) -> dict:
    return {"chunks": retrieve(state["search_query"], settings.top_k)}


def build_context_node(state: AgentState) -> dict:
    return {"context": build_context(state["chunks"])}


def generate_node(state: AgentState) -> dict:
    return {"answer": generate_answer(state["question"], state["context"])}


def output_guard_node(state: AgentState) -> dict:
    result = check_output(state["question"], state["context"], state["answer"])
    return {"guardrail_results": [result]}


def refuse_node(state: AgentState) -> dict:
    failed_guard = state["guardrail_results"][-1].guard
    return {"answer": REFUSALS[failed_guard]}


def route_after_guard(state: AgentState) -> str:
    return "continue" if state["guardrail_results"][-1].allowed else "refuse"
