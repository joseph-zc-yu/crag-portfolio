import json
import os
import re
from typing import Any, Literal, TypedDict

from ddgs import DDGS
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from pdf_processor import DocumentChunk
from vector_store import LightVectorStore


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("GLM_MODEL", "glm-4.7-flash"),
        api_key=os.getenv("ZAI_API_KEY"),
        base_url=os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4"),
        temperature=0.1,
    )


def parse_json_safe(raw: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        return json.loads(match.group(0))
    return {}


class CRAGState(TypedDict):
    session_id: str
    query: str
    chat_history: list[dict[str, str]]
    retrieved_chunks: list[dict[str, Any]]
    relevance: Literal["high", "insufficient"]
    relevant_chunk_ids: list[str]
    web_queries: list[str]
    web_results: list[dict[str, str]]
    draft_context: str
    final_answer: str
    citations: list[dict[str, Any]]


def build_crag_graph(store: LightVectorStore):
    llm = get_llm()

    # [1. Retrieve Node]
    def retrieve_node(state: CRAGState) -> dict[str, Any]:
        hits = store.similarity_search(state["query"], top_k=4)
        formatted = [
            {
                "chunk_id": chunk.chunk_id,
                "section_title": chunk.section_title,
                "page_number": chunk.page_number,
                "text": chunk.text,
                "bboxes_by_page": chunk.bboxes_by_page,
                "score": round(score, 4),
            }
            for chunk, score in hits
        ]
        return {"retrieved_chunks": formatted}

    # [2. Grade Node]
    def grade_node(state: CRAGState) -> dict[str, Any]:
        chunks_text = "\n\n".join(
            f"ID: {c['chunk_id']} | Section: {c['section_title']} (Page {c['page_number']})\n{c['text']}"
            for c in state["retrieved_chunks"]
        )
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in state.get("chat_history", [])[-4:]
        )

        prompt = f"""You are a strict retrieval evaluator for a Corrective RAG system.
Conversation History:
{history_text or "None"}

User Query: {state['query']}

Retrieved PDF Sections:
{chunks_text}

Task:
1. Determine if the retrieved PDF sections contain SUFFICIENT, grounded information to fully answer the user query.
2. List the exact chunk IDs that are relevant (e.g., ["c_1", "c_3"]).
3. If the document lacks info, is outdated, or needs outside facts, set relevance to "insufficient" and provide 1-2 concise web search queries.

Respond ONLY in valid JSON:
{{
  "relevance": "high" or "insufficient",
  "relevant_chunk_ids": ["..."],
  "web_queries": ["..."]
}}"""

        resp = llm.invoke(prompt)
        data = parse_json_safe(str(resp.content))
        relevance = "high" if data.get("relevance") == "high" else "insufficient"
        relevant_ids = data.get("relevant_chunk_ids", [])
        web_queries = data.get("web_queries", [])
        if relevance == "insufficient" and not web_queries:
            web_queries = [state["query"]]

        return {
            "relevance": relevance,
            "relevant_chunk_ids": relevant_ids,
            "web_queries": web_queries[:2],
        }

    def route_after_grade(state: CRAGState) -> str:
        return "generate_node" if state["relevance"] == "high" else "web_search_fallback"

    # [3. Generate Node] (High relevance path: prepare refined PDF context)
    def generate_node(state: CRAGState) -> dict[str, Any]:
        selected = [
            c
            for c in state["retrieved_chunks"]
            if not state["relevant_chunk_ids"] or c["chunk_id"] in state["relevant_chunk_ids"]
        ]
        context = "\n\n".join(
            f"[Doc:{c['chunk_id']} | {c['section_title']}, p.{c['page_number']}]\n{c['text']}"
            for c in selected
        )
        return {"draft_context": context, "web_queries": [], "web_results": []}

    # [4. Web Search Fallback]
    def web_search_fallback(state: CRAGState) -> dict[str, Any]:
        results: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        for q in state.get("web_queries", []):
            try:
                with DDGS() as ddgs:
                    hits = list(ddgs.text(q, max_results=3))
                for h in hits:
                    url = h.get("href", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        results.append(
                            {
                                "id": f"web_{len(results) + 1}",
                                "query": q,
                                "title": h.get("title", "Web Result"),
                                "url": url,
                                "snippet": h.get("body", ""),
                            }
                        )
            except Exception as exc:
                results.append(
                    {
                        "id": f"web_err_{len(results) + 1}",
                        "query": q,
                        "title": "Search Temporarily Rate-Limited",
                        "url": "",
                        "snippet": f"DuckDuckGo fallback note: {exc}",
                    }
                )
        return {"web_results": results}

    # [Append Web Results to State]
    def append_web_results(state: CRAGState) -> dict[str, Any]:
        doc_part = "\n\n".join(
            f"[Doc:{c['chunk_id']} | {c['section_title']}, p.{c['page_number']}]\n{c['text']}"
            for c in state["retrieved_chunks"]
            if not state["relevant_chunk_ids"] or c["chunk_id"] in state["relevant_chunk_ids"]
        )
        web_part = "\n\n".join(
            f"[Web:{w['id']} | {w['title']} ({w['url']})]\n{w['snippet']}"
            for w in state.get("web_results", [])
        )
        combined = f"--- PDF CONTEXT ---\n{doc_part or 'No strongly matching PDF sections.'}\n\n--- WEB CONTEXT ---\n{web_part}"
        return {"draft_context": combined}

    # [Final Generation]
    def final_generation(state: CRAGState) -> dict[str, Any]:
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in state.get("chat_history", [])[-6:]
        )

        prompt = f"""You are a precise assistant answering questions strictly from grounded context.
Conversation History:
{history_text or "None"}

Grounded Context:
{state['draft_context']}

User Question: {state['query']}

Instructions:
1. Answer clearly and concisely using ONLY facts from the Grounded Context above.
2. When citing the PDF, embed inline tags like [Doc:c_1]. When citing web sources, embed [Web:web_1].
3. Also return the list of cited Doc chunk IDs so the UI can highlight them on the PDF.

Respond ONLY in valid JSON:
{{
  "answer": "Your grounded markdown answer with [Doc:...] and [Web:...] citations...",
  "cited_chunk_ids": ["c_1"]
}}"""

        resp = llm.invoke(prompt)
        data = parse_json_safe(str(resp.content))
        answer = data.get("answer") or str(resp.content)
        cited_ids = set(data.get("cited_chunk_ids", []))

        # Fallback: extract [Doc:c_X] tags from answer text if JSON omitted cited_chunk_ids
        if not cited_ids:
            cited_ids = set(re.findall(r"\[Doc:(c_\d+)\]", answer))

        citations = [
            {
                "chunk_id": c["chunk_id"],
                "section_title": c["section_title"],
                "page_number": c["page_number"],
                "excerpt": c["text"][:240] + ("..." if len(c["text"]) > 240 else ""),
                "bboxes_by_page": c["bboxes_by_page"],
            }
            for c in state["retrieved_chunks"]
            if c["chunk_id"] in cited_ids
        ]

        return {"final_answer": answer, "citations": citations}

    builder = StateGraph(CRAGState)
    builder.add_node("retrieve_node", retrieve_node)
    builder.add_node("grade_node", grade_node)
    builder.add_node("generate_node", generate_node)
    builder.add_node("web_search_fallback", web_search_fallback)
    builder.add_node("append_web_results", append_web_results)
    builder.add_node("final_generation", final_generation)

    builder.add_edge(START, "retrieve_node")
    builder.add_edge("retrieve_node", "grade_node")
    builder.add_conditional_edges(
        "grade_node",
        route_after_grade,
        {
            "generate_node": "generate_node",
            "web_search_fallback": "web_search_fallback",
        },
    )
    builder.add_edge("generate_node", "final_generation")
    builder.add_edge("web_search_fallback", "append_web_results")
    builder.add_edge("append_web_results", "final_generation")
    builder.add_edge("final_generation", END)

    return builder.compile()