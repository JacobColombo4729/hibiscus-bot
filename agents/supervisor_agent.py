"""
This module defines the SupervisorAgent, which acts as the central orchestrator for the Hibiscus Bot.

The SupervisorAgent is responsible for receiving user input, coordinating with specialized agents 
(e.g., NutritionAgent, FitnessAgent), and generating a final response. It uses a large language 
model (LLM) to interpret user queries and decide the next steps.
"""
import os
import json
from typing import List, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from utils.prompts import supervisor_prompt_template
from .nutrition_agent import NutritionAgent
from .fitness_agent import FitnessAgent
from .general_agent import GeneralAgent  
from .state import State
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from utils.embeddings import retrieve_relevant_chunks, sample_data_collection

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class SupervisorAgent:
    """
    The main agent that orchestrates the workflow of the Hibiscus Bot.

    This class initializes the LLM and the specialized agents, and its `run` method
    is the main entry point for processing user input. It retrieves relevant information
    from a vector database and uses the LLM to generate a response.
    """
    def __init__(self):
        """
        Initializes the SupervisorAgent.

        This involves setting up the primary LLM and instantiating the specialized agents
        (NutritionAgent, FitnessAgent, and GeneralAgent), passing the LLM to them.
        """
        # The supervisor is the single source of truth for the LLM
        self.llm_model = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1,
        )

        # Instantiate the specialist agents, injecting the shared LLM
        self.nutrition_agent = NutritionAgent(self.llm_model)
        self.fitness_agent = FitnessAgent(self.llm_model)
        self.general_agent = GeneralAgent(self.llm_model)

    def run(self, user_input: str) -> str:
        """
        Processes the user's input and returns a response.

        This is the main entry point for the supervisor. It takes the user's input,
        retrieves relevant context from the knowledge base, and uses the LLM to generate
        a helpful response.

        Args:
            user_input (str): The input string from the user.

        Returns:
            str: The generated response to be sent to the user.
        """
        print("---SUPERVISOR: DECIDING NEXT STEP---")
        
        # Retrieve relevant chunks
        relevant_chunks = retrieve_relevant_chunks(user_input, sample_data_collection, 3)
        chunks_text = "\n".join([chunk for chunk, id in relevant_chunks])
        
        # Create a prompt for reply
        reply_prompt = f"Use the following context to answer the query: {chunks_text}\nQuery: {user_input}"
        
        # Invoke the LLM to generate reply
        response = self.llm_model.invoke(reply_prompt)
        reply = response.content
        
        return reply

    # def workflow(self):
    #     """
    #     Builds and compiles the LangGraph workflow for the agentic system.
    #
    #     This method sets up a stateful graph that defines the flow of conversation
    #     between the supervisor and the specialized agents. It is currently commented out
    #     but can be used to implement a more complex, multi-agent workflow.
    #
    #     Returns:
    #         A compiled LangGraph workflow.
    #     """
    #     # Use a StateGraph for built-in state management
    #     graph_builder = StateGraph(State)

    #     graph_builder.add_node("supervisor", self.run)
    #     graph_builder.add_node("nutrition_node", self.nutrition_agent.run)
    #     graph_builder.add_node("fitness_node", self.fitness_agent.run)
    #     graph_builder.add_node("general_node", self.general_agent.run)
        
    #     graph_builder.set_entry_point("supervisor")
        
    #     graph_builder.add_conditional_edges(
    #         "supervisor",
    #         lambda state: state["next"],
    #         {
    #             "nutrition_node": "nutrition_node",
    #             "fitness_node": "fitness_node",
    #             "general_node": "general_node",
    #             "__end__": END
    #         }
    #     )
        
    #     graph_builder.add_edge("nutrition_node", "supervisor")
    #     graph_builder.add_edge("fitness_node", "supervisor")
    #     graph_builder.add_edge("general_node", "supervisor")

    #     graph_builder.set_entry_point("supervisor")

    #     return graph_builder.compile()



