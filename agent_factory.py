# agent_factory.py

import chainlit as cl
from hh_about_agent import load_support_docs, build_retriever, create_agent

def get_gen_info_agent(username):
    session = cl.user_session
    if not session.get("retriever"):
        support_docs = load_support_docs("about_hh_docs")
        session.set("retriever", build_retriever(support_docs))
    if not session.get("user_agents"):
        session.set("user_agents", {})
    user_agents = session.get("user_agents")
    if username not in user_agents:
        user_agents[username] = create_agent(session.get("retriever"))
        session.set("user_agents", user_agents)
    return user_agents[username]