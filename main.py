"""
This module serves as the main entry point for the Hibiscus Bot, a personal wellness assistant.

It uses FastAPI to create a web server that exposes a chat endpoint for interacting with the bot.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from agents.supervisor_agent import SupervisorAgent

app = FastAPI(
    title="Hibiscus Bot",
    description="A personal wellness assistant.",
    version="1.0.0",
)

class ChatMessage(BaseModel):
    message: str
    history: str = ""

class ChatResponse(BaseModel):
    reply: str

supervisor = SupervisorAgent()

@app.post("/chat", response_model=ChatResponse)
async def chat(item: ChatMessage):
    """
    Handles incoming chat messages from the user.

    This endpoint takes a user's message, passes it to the SupervisorAgent's `run` method,
    and returns the agent's reply.

    Args:
        item (ChatMessage): The request body containing the user's message and optional history.

    Returns:
        ChatResponse: A response object containing the bot's reply.
    """
    reply = supervisor.run(item.message, item.history)
    return ChatResponse(reply=reply)

@app.get("/")
async def root():
    """
    Root endpoint that returns a welcome message.
    """
    return {"message": "Welcome to the Hibiscus Bot API!"}
