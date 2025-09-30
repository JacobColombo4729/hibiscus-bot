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
        self.general_agent = GeneralAgent(self.llm_model)

    def run(self, user_input: str, history: str = "") -> str:
        """
        Processes the user's input and returns a response.

        This is the main entry point for the supervisor. It takes the user's input,
        uses the supervisor prompt template to determine which agent should handle the query,
        and routes the query accordingly.

        Args:
            user_input (str): The input string from the user.
            history (str): The conversation history (optional).

        Returns:
            str: The generated response to be sent to the user.
        """
        print("---SUPERVISOR: DECIDING NEXT STEP---")
        
        # Use the supervisor prompt template to determine routing
        prompt = supervisor_prompt_template.format_messages(
            history=history,
            input=user_input
        )
        
        # Get the routing decision from the LLM
        response = self.llm_model.invoke(prompt)
        routing_decision = response.content
        
        # Parse the JSON response to determine which agents to use
        try:
            import json
            decision = json.loads(routing_decision)
            agents = decision.get("agents", [])
            reasoning = decision.get("reasoning", "")
            
            print(f"Routing decision: {reasoning}")
            print(f"Selected agents: {agents}")
            
            # If no agents are selected, handle as general query
            if not agents:
                return self._handle_general_query(user_input)
            
            # Route to appropriate agents
            return self._route_to_agents(user_input, agents)
            
        except json.JSONDecodeError:
            print("Failed to parse routing decision, handling as general query")
            return self._handle_general_query(user_input)
    
    def _handle_general_query(self, user_input: str) -> str:
        """
        Handles general queries that don't require specialized agents.
        
        Args:
            user_input (str): The user's input.
            
        Returns:
            str: The response from the general agent.
        """
        return self.general_agent.run(user_input)
    
    def _route_to_agents(self, user_input: str, agents: List[str]) -> str:
        """
        Routes the query to the appropriate specialized agents.
        
        Args:
            user_input (str): The user's input.
            agents (List[str]): List of agent names to handle the query.
            
        Returns:
            str: The combined response from the selected agents.
        """
        responses = []
        
        for agent_name in agents:
            if "Meal Planning" in agent_name or "Nutrition" in agent_name:
                response = self.nutrition_agent.run(user_input)
                responses.append(f"**Nutrition Expert:**\n{response}")
            elif "General" in agent_name:
                response = self.general_agent.run(user_input)
                responses.append(f"**General Assistant:**\n{response}")
            else:
                # Fallback to general agent for any unrecognized agent names
                response = self.general_agent.run(user_input)
                responses.append(f"**General Assistant:**\n{response}")
        
        # Combine responses
        if len(responses) == 1:
            return responses[0]
        else:
            return "\n\n".join(responses)

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



