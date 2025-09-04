"""
This module serves as the main entry point for the Hibiscus Bot, a personal wellness assistant.

It uses the chainlit library to create a chat interface and handles the chat logic by 
interacting with a SupervisorAgent that coordinates various specialized agents.
"""
import chainlit as cl
from langchain_core.messages import HumanMessage
from agents.supervisor_agent import SupervisorAgent

@cl.on_chat_start
async def main():
    """
    Initializes and sets up the chat session when a user starts a new chat.

    This function creates a new `SupervisorAgent`, stores it in the user's session, 
    and sends a welcome message to the user.
    """
    supervisor = SupervisorAgent()
    cl.user_session.set("supervisor", supervisor)
    await cl.Message(content="Hello! I am Hibiscus Bot, your personal wellness assistant. How can I help you today?").send()

@cl.on_message
async def on_message(message: cl.Message):
    """
    Handles incoming messages from the user.

    This function retrieves the `SupervisorAgent` from the user's session, passes the 
    user's message to the agent's `run` method, and sends the agent's reply back to the user.

    Args:
        message (cl.Message): The message object from the user.
    """
    supervisor = cl.user_session.get("supervisor")
    reply = await cl.make_async(supervisor.run)(message.content)
    await cl.Message(content=reply).send()

