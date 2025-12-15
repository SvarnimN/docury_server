from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.documents import Document
from app.services.vector import VectorService
from app.config import Configs
from app.utils import get_prompt_templates, format_docs
from typing import Dict, Any, List
from loguru import logger


class RAGService:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

        self.vector_retriever = VectorService().get_retriever()
        self.wiki_retriever = WikipediaRetriever(
            top_k_results=2,
            doc_content_chars_max=3000,
        )

        self.prompt_template = get_prompt_templates()

        self.rag_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.prompt_template["system"]),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        ).partial(context="")

        self.condense_chain = (
            ChatPromptTemplate.from_template(self.prompt_template["question"])
            | self.llm
            | StrOutputParser()
        )

        def get_standalone_question(x: Dict[str, Any]) -> str:
            if x.get("chat_history"):
                return self.condense_chain.invoke(x)
            return x["question"]

        inputs = RunnablePassthrough.assign(
            standalone_question=RunnableLambda(get_standalone_question)
        )

        def is_context_sufficient(docs: List[Document], min_chars: int = 300) -> bool:
            if not docs:
                return False
            return sum(len(d.page_content) for d in docs) >= min_chars

        def hybrid_retrieve(x: Dict[str, Any]) -> List[Document]:
            question = x["standalone_question"]

            logger.info("Retrieving from vector store")
            vector_docs = self.vector_retriever.invoke(question)

            if is_context_sufficient(vector_docs):
                logger.info("Vector context sufficient")
                return vector_docs

            logger.info("Vector context insufficient, adding Wikipedia")
            wiki_docs = self.wiki_retriever.invoke(question)

            return vector_docs + wiki_docs

        retrieval_chain = {
            "context": RunnableLambda(hybrid_retrieve) | format_docs,
            "question": RunnableLambda(lambda x: x["question"]),
            "chat_history": RunnableLambda(lambda x: x["chat_history"]),
        }

        rag_chain = (
            inputs
            | retrieval_chain
            | self.rag_prompt
            | self.llm
            | StrOutputParser()
        )

        self.chain_with_history = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )

    def get_session_history(self, session_id: str):
        return RedisChatMessageHistory(
            session_id=session_id,
            url=Configs.REDIS_URL,
        )

    def ask_question(self, question: str, session_id: str):
        config = {"configurable": {"session_id": session_id}}
        return self.chain_with_history.invoke(
            {"question": question},
            config=config,
        )
