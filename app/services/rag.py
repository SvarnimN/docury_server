from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from app.services.vector import VectorService
from app.config import Configs
from app.utils import get_prompt_templates, format_docs
from typing import Dict, Any

class RAGService:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
        self.retriever = VectorService().get_retriever()
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
        
        def _get_standalone_question(input_dict: Dict[str, Any]):
            return self.condense_chain.invoke(input_dict)

        original_question_chain = RunnableLambda(lambda x: x["question"])

        def check_history(x: Dict[str, Any]) -> bool:
            return bool(x.get("chat_history"))

        standalone_question_chain = RunnableBranch(
            (check_history, RunnableLambda(_get_standalone_question)),
            original_question_chain
        )

        _inputs = RunnablePassthrough.assign(
            standalone_question=standalone_question_chain
        )

        retrieval_chain = {
            "context": RunnableLambda(lambda x: x["standalone_question"])
                       | self.retriever
                       | format_docs,
            "question": RunnableLambda(lambda x: x["question"]),
            "chat_history": RunnableLambda(lambda x: x["chat_history"])
        }
        
        rag_chain = _inputs | retrieval_chain | self.rag_prompt | self.llm | StrOutputParser()

        self.chain_with_history = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )
    
    def get_session_history(self, session_id: str):
        return RedisChatMessageHistory(session_id=session_id, url=Configs.REDIS_URL)

    def ask_question(self, question: str, session_id: str):
        config = {"configurable": {"session_id": session_id}}
        
        response = self.chain_with_history.invoke({"question": question}, config=config)
        
        return response
