from langchain_core.documents import Document
from typing import List, Dict


def get_prompt_templates() -> Dict[str, str]:
    system_prompt = """
    You are a helpful AI assistant.

    Answer the user's question using ONLY the provided document context.

    Rules:
    - Greet users when appropriate.
    - For follow-up questions, refer to chat history.
    - Do NOT use prior knowledge or general knowledge.
    - Every factual statement MUST include a citation: <source='filename', page=number>
    - If the answer is not in the documents, respond:
      "I cannot find this information in the provided documents."
    - If the question is unrelated, respond:
      "The provided question is irrelevant to the given documents."
    - Convert any Markdown or HTML into plain text.

    Set is_answer_found True if answer is in documents; otherwise False.
    Document Context:
    {context}
    """

    question_prompt = """
    Rewrite a follow-up question to be understood independently.
    Do NOT answer or add information. Return ONLY the rewritten question.

    Chat History:
    {chat_history}

    Follow-up Question:
    {question}

    Standalone Question:
    """

    return {"system": system_prompt.strip(), "question": question_prompt.strip()}


def format_docs(docs: List[Document]) -> str:
    """
    Convert a list of Documents into a formatted string for LLM input.
    Includes content and metadata (source and page).
    """
    formatted = []

    for i, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        source_file = str(metadata.get("source", "unknown")).split("/")[-1]
        page = metadata.get("page", "unknown")

        formatted.append(
            f"--- Snippet {i} ---\n"
            f"{doc.page_content.strip()}\n"
            f"<source='{source_file}', page={page}>"
        )

    return "\n\n".join(formatted)
