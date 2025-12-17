from typing import Dict, Any, List
from pydantic import BaseModel, Field
from loguru import logger

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.documents import Document
from langchain_cohere import ChatCohere
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_aws import ChatBedrock
from app.services.vector import VectorService
from app.config import Configs, AWSConfigs
from app.utils import get_prompt_templates, format_docs


class LLMResponse(BaseModel):
    answer: str = Field(description="Relevant answers based on the provided context.")
    is_found: bool = Field(description="Boolean to check if the answer is found.")


class RAGService:
    def __init__(self, store_dir: str = Configs.VECTOR_DIR):
        self.llm = ChatBedrock(
            region_name=AWSConfigs.REGION_NAME,
            model_id=AWSConfigs.CHAT_MODEL_ID,
            model_kwargs=AWSConfigs.MODEL_KWARGS,
        )

        self.vector_retriever = VectorService(store_dir=store_dir).get_retriever()
        self.wiki_retriever = WikipediaRetriever(top_k_results=5, doc_content_chars_max=3000)
        self.prompt_template = get_prompt_templates()

        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompt_template["system"]),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        self.chain = (
            ChatPromptTemplate.from_template(self.prompt_template["question"])
            | self.llm
            | StrOutputParser()
        )

        self.inputs = RunnablePassthrough.assign(
            standalone_question=RunnableLambda(self._get_standalone_question)
        )

    def _get_standalone_question(self, x: Dict[str, Any]) -> str:
        return self.chain.invoke(x) if x.get("chat_history") else x["question"]

    def retriever(self, x: Dict[str, Any], retriever_type: str) -> List[Document]:
        question = x["standalone_question"]
        if retriever_type == "vector":
            logger.info("Searching vector store")
            return self.vector_retriever.invoke(question)
        else:
            logger.info("Searching Wikipedia")
            return self.wiki_retriever.invoke(question)

    def get_final_chain(self, retriever_type: str):
        retrieval_chain = {
            "context": RunnableLambda(lambda x: self.retriever(x, retriever_type))
            | RunnableLambda(lambda x: format_docs(x, retriever_type)),
            "question": RunnableLambda(lambda x: x["question"]),
            "chat_history": RunnableLambda(lambda x: x["chat_history"]),
        }

        rag_chain = (
            self.inputs
            | retrieval_chain
            | self.rag_prompt
            | self.llm.with_structured_output(LLMResponse)
        )

        return RunnableWithMessageHistory(
            rag_chain,
            self._get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )

    def _get_session_history(self, session_id: str):
        return RedisChatMessageHistory(session_id=session_id, url=Configs.REDIS_URL)

    def ask_question(self, question: str, session_id: str) -> str:
        config = {"configurable": {"session_id": session_id}}

        response = self.get_final_chain("vector").invoke({"question": question}, config=config)
        if response.is_found:
            return response.answer

        response = self.get_final_chain("wiki").invoke({"question": question}, config=config)
        return response.answer
