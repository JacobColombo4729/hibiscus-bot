"""
This module defines the GeneralAgent, a versatile agent that handles general wellness queries 
and conversations that do not fall into the specialized domains of nutrition or fitness.

The GeneralAgent is designed to be a "catch-all" agent in the agentic system, providing helpful 
advice and engaging in friendly conversation.
"""
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_groq import ChatGroq

class GeneralAgent:
    """
    A general-purpose agent for handling non-specialized wellness queries.

    This agent is initialized with a language model and provides helpful responses
    for general wellness questions and casual conversation.
    """
    def __init__(self, llm_model: ChatGroq):
        """
        Initializes the GeneralAgent.

        Args:
            llm_model (ChatGroq): The language model that the agent will use to 
                                 process queries and generate responses.
        """
        self.llm_model = llm_model
        
        self.system_prompt = (
            "You are a general wellness assistant. Handle queries that do not fall into specific categories "
            "like nutrition or fitness. Provide helpful, general advice and engage in friendly conversation. "
            "Be warm, supportive, and encouraging in your responses."
        )

    def run(self, user_input: str) -> str:
        """
        Executes the main logic of the GeneralAgent.

        This method processes the user input and returns a response string.

        Args:
            user_input (str): The user's input string.

        Returns:
            str: The agent's response as a string.
        """
        print("---EXECUTING GENERAL AGENT---")
        
        # Create a simple prompt with the system message and user input
        prompt = f"{self.system_prompt}\n\nUser: {user_input}\nAssistant:"
        
        # Invoke the LLM directly
        response = self.llm_model.invoke(prompt)
        
        return response.content
