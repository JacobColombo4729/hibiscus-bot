"""
This module defines the NutritionAgent, a specialized agent focused on providing 
nutrition-related information and answering frequently asked questions.

The NutritionAgent is designed to be part of a larger agentic system, where it can be 
invoked by a supervisor agent to handle specific user queries about nutrition and diet.
"""
from langchain_groq import ChatGroq
from utils.embeddings import retrieve_relevant_chunks, sample_data_collection

class NutritionAgent:
    """
    A specialized agent that handles nutrition-related queries.

    This agent is initialized with a language model and provides specialized responses
    for nutrition and diet-related questions using RAG (Retrieval Augmented Generation).
    """
    def __init__(self, llm_model: ChatGroq):
        """
        Initializes the NutritionAgent.

        Args:
            llm_model (ChatGroq): The language model that the agent will use to 
                                 process queries and generate responses.
        """
        self.llm_model = llm_model
        
        self.system_prompt = (
            "You are a specialized nutrition expert. Provide accurate, evidence-based information "
            "about nutrition, diet, meal planning, and healthy eating habits. Use the provided context "
            "to give specific, helpful advice. Always prioritize user safety and recommend consulting "
            "healthcare professionals for serious health concerns."
        )

    def run(self, user_input: str) -> str:
        """
        Executes the main logic of the NutritionAgent.

        This method processes the user input and returns a response string.

        Args:
            user_input (str): The user's input string.

        Returns:
            str: The agent's response as a string.
        """
        print("---EXECUTING NUTRITION AGENT---")
        
        # Retrieve relevant chunks
        relevant_chunks = retrieve_relevant_chunks(user_input, sample_data_collection, 3)
        chunks_text = "\n".join([chunk for chunk, id in relevant_chunks])
        
        # Create a comprehensive prompt with system message, context, and user query
        prompt = f"{self.system_prompt}\n\nContext from knowledge base:\n{chunks_text}\n\nUser Query: {user_input}\n\nNutrition Expert Response:"
        
        # Invoke the LLM to generate reply
        response = self.llm_model.invoke(prompt)
        
        return response.content
