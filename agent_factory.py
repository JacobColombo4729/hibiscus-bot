# agent_factory.py

import chainlit as cl
from gen_info_agent import load_support_docs, build_retriever, create_agent

# def get_user_agent(username, orders):
def get_gen_info_agent(username):
    session = cl.user_session
    if "retriever" not in session:
        support_docs = load_support_docs("docs")
        session["retriever"] = build_retriever(support_docs)
    if "user_agents" not in session:
        session["user_agents"] = {}
    if username not in session["user_agents"]:
        # session["user_agents"][username] = create_agent(session["retriever"], orders)
        session["user_agents"][username] = create_agent(session["retriever"])
    return session["user_agents"][username]