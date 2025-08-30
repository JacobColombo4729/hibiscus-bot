# (Agent logic: RAG + Order Lookup tool)

# customer_support_agent.py

import os
import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
import boto3
from langchain_aws import ChatBedrock

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
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory="chroma_db")
    return vectordb.as_retriever()

def rag_tool_func(query, retriever):
    results = retriever.get_relevant_documents(query)
    if not results:
        return "Sorry, I couldn't find any information in our knowledge base."
    return "\n\n".join([f"From {r.metadata['source']}: {r.page_content.strip()}" for r in results])

# def order_lookup_tool_func(order_id, orders):
#     print("=====Printing order details New=====")
#     print(order_id)
#     order_id = order_id.replace("```","").strip()
#     print(order_id)
#     print(orders)
#     for o in orders:
#         print(o)
#         print(order_id)
#         print(o["order_id"] ,order_id )
#         if o["order_id"] == order_id:
#             print("Order found")
#             return (f"Order ID: {o['order_id']}\n"
#                     f"Item: {o['item']}\n"
#                     f"Status: {o['status']}\n"
#                     f"Date: {o['date']}")
#     return f"Sorry, I could not find any order with ID {order_id}."

# def create_agent(retriever, orders):
def create_agent(retriever):
    rag_tool = Tool(
        name="KnowledgeBaseSearch",
        func=lambda query: rag_tool_func(query, retriever),
        description="Use this tool to answer user questions about general information on HH"
    )
    # order_lookup_tool = Tool(
    #     name="OrderLookup",
    #     func=lambda order_id: order_lookup_tool_func(order_id, orders),
    #     description="Use this tool to look up the status of a customer's order by order ID. The user must provide an order ID."
    # )

    memory = ConversationBufferWindowMemory(k=15, memory_key="history")
    cl.user_session.set("memory", memory)

    llm = ChatBedrock(
        client=boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
        ),
        model_id="meta.llama3-8b-instruct-v1:0",
        model_kwargs={"temperature": 0.4, "max_gen_len": 1024,},
    )
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = initialize_agent(
        # tools=[order_lookup_tool, rag_tool],
        tools=[rag_tool],
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=False,
        handle_parsing_errors=True
    )
    return agent