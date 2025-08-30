# customer_support_agent.py

import os
import chainlit as cl
# from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_aws.embeddings import BedrockEmbeddings

def load_support_docs(docs_dir="gen_info_docs"):
    all_docs = []
    for filename in os.listdir(docs_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(docs_dir, filename)
            loader = TextLoader(file_path)
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = filename
            all_docs.extend(docs)
    return all_docs

def build_retriever(docs):
    embedding_function = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        region_name="us-east-1"
    )
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding_function,
        persist_directory="chroma_db",
        collection_name="{} collection"
    )
    return vectordb.as_retriever()

def rag_tool_func(query, retriever):
    results = retriever.get_relevant_documents(query)
    if not results:
        return "Sorry, I couldn't find any information in our knowledge base."
    return "\n\n".join([f"From {r.metadata['source']}: {r.page_content.strip()}" for r in results])

# def create_agent(retriever):
    # rag_tool = Tool(
    #     name="KnowledgeBaseSearch",
    #     func=lambda query: rag_tool_func(query, retriever),
    #     description="Use this tool to answer general questions about Hibiscus Health"
    # )
    # llm = ChatBedrock(
    #     client=boto3.client(
    #         "bedrock-runtime",
    #         region_name="us-east-1",
    #     ),
    #     model_id="meta.llama3-8b-instruct-v1:0",
    #     model_kwargs={"temperature": 0.4, "max_gen_len": 1024,},
    # )

    # cl.user_session.set("llm", llm)
    # memory = ConversationBufferWindowMemory(k=15, memory_key="history")
    # cl.user_session.set("memory", memory)
    # agent = initialize_agent(
    #     tools=[rag_tool],
    #     llm=llm,
    #     agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    #     memory=memory,
    #     verbose=False,
    #     handle_parsing_errors=True
    # )
    # return agent

from agent_tools import load_support_docs, build_retriever, create_agent

def get_agent(username):
    session = cl.user_session
    if not session.get("retriever"):
        # Update load_support_docs to actually load documents
        support_docs = load_support_docs()
        session.set("retriever", build_retriever(support_docs))
    if not session.get("user_agents"):
        session.set("user_agents", {})
    user_agents = session.get("user_agents")
    if username not in user_agents:
        user_agents[username] = create_agent(session.get("retriever"))
        session.set("user_agents", user_agents)
    return user_agents[username]