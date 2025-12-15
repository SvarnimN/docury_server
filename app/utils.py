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


def format_docs(docs: List[Document], retriever_type: str = "vector") -> str:
    """
    Convert a list of Documents into a formatted string for LLM input.
    Adds important metadata for each document, depending on the retriever type.
    """
    formatted_snippets = []

    for idx, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}

        source = metadata.get("source", "unknown")
        page = metadata.get("page", "unknown")
        title = metadata.get("title", "")
        description = metadata.get("description", "")
        language = metadata.get("language", "")
        chunk_index = metadata.get("chunk_index", "")

        source_file = source.split("/")[-1] if isinstance(source, str) else str(source)

        if retriever_type.lower() == "vector":
            meta_str = (
                f"<source='{source_file}', title='{title}', description='{description}', "
                f"page='{page}', chunk_index='{chunk_index}', language='{language}'>"
            )
            snippet_label = f"--- Vector Snippet {idx} ---"
        elif retriever_type.lower() == "wiki":
            meta_str = (
                f"<source='{source_file}', title='{title}', description='{description}', page='{page}'>"
            )
            snippet_label = f"--- Wiki Snippet {idx} ---"
        else:
            meta_str = f"<source='{source_file}', page='{page}'>"
            snippet_label = f"--- Snippet {idx} ---"

        formatted_snippets.append(
            f"{snippet_label}\n{doc.page_content.strip()}\n{meta_str}"
        )

    return "\n\n".join(formatted_snippets)
