from langchain_core.documents import Document
from typing import List, Dict


def get_prompt_templates() -> Dict[str, str]:
    SYSTEM = """
    You are a helpful AI assistant.

    You MUST answer the user's question using ONLY the information provided
    in the document context below.

    Rules:
    - Provide greetings for user greetings.
    - Do NOT use prior knowledge or general knowledge.
    - Every factual statement MUST include a citation.
    - Use the citation format exactly as shown:
    <source='filename', page=number>
    - If the answer is not present in the documents, say:
    "I cannot find this information in the provided documents."
    - If the question is unrelated to the documents, say:
    "The provided question is irrelevant to the given documents."
    - Convert any Markdown or HTML into plain text in your response.

    Set is_answer_found to True if the relevant answer is found in
    the provided documents. Otherwise, set is_answer_found to False.
    Similary, set answer to the correct answer string. Otherwise, set
    answer to 'No relevant answer found in the provided document'.

    Document Context:
    {context}
    """
    
    QUESTION = """
    Given the following conversation and a follow-up question,
    rewrite the follow-up question so that it can be understood
    independently without the chat history.

    Do NOT answer the question.
    Do NOT add new information.
    Return ONLY the rewritten question.

    Chat History:
    {chat_history}

    Follow-up Question:
    {question}

    Standalone Question:
    """

    return {
        "system": SYSTEM.strip(),
        "question": QUESTION.strip(),
    }


def format_docs(docs: List[Document]) -> str:
    formatted_snippets = []

    for idx, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}

        source = metadata.get("source", "unknown")
        page = metadata.get("page", "unknown")

        source_file = source.split("/")[-1] if isinstance(source, str) else "unknown"

        formatted_snippets.append(
            f"--- Snippet {idx} ---\n"
            f"{doc.page_content.strip()}\n"
            f"<source='{source_file}', page={page}>"
        )

    return "\n\n".join(formatted_snippets)
