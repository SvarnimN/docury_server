from langchain_core.documents import Document
from typing import List, Dict

def get_prompt_templates() -> Dict[str, str]:
    SYSTEM = """
        You are a helpful AI assistant that answers user questions using your
        general knowledge and, where relevant, content from provided documents. 
        When you use the documents, quote the source metadata in the format
        (Source: '{{source_meta}}', Page: {{page_meta}}).

        Use the following extracted document context to answer the question.
        If snippets don't contain the answer, say you cannot find the information
        in the documents and then answer using your general knowledge.
        If the asked question is not relevant to the context, simply say that
        the provided question is irrelevant.

        Context:
        {context}

        Also, clean markdown and HTML codes to plain texts when providing your response.
    """

    QUESTION = """
        Given the following conversation and a follow-up question, rephrase the follow-up question
        to be a standalone search query. Do not answer the question, just rephrase it.
        If the follow-up question is already a standalone question, return it as is.

        Chat history:
        {chat_history}

        Follow Up Question: {question}

        Standalone Search Query:
    """

    return {
        "system": SYSTEM,
        "question": QUESTION
    }


def format_docs(docs: List[Document]) -> str:
    formatted_snippets = []
    
    for i, doc in enumerate(docs):
        source_meta = doc.metadata.get("source", "unknown file")
        page_meta = doc.metadata.get("page", "unknown")

        source_file = source_meta.split("/")[-1]
        
        formatted_snippets.append(
            f"--- Snippet {i+1} ---\n"
            f"Content: {doc.page_content}\n"
            f"Citation: <source='{source_file}', page={page_meta}>\n"
        )
    
    return "\n\n".join(formatted_snippets)
