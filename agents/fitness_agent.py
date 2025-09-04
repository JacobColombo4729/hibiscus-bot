"""
This module defines the FitnessAgent, a specialized agent for handling tasks related to 
fitness and appointments, such as scheduling, canceling, or rescheduling.

The FitnessAgent is intended to be part of a larger agentic system, where it can be 
invoked by a supervisor agent to manage fitness-related user queries.
"""
from typing import Literal
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from .state import State

class FitnessAgent:
    """
    A specialized agent that handles fitness and appointment-related queries.

    This agent is initialized with a language model and is designed to work within
    a LangGraph workflow. It uses a predefined system prompt to guide its behavior
    and can be equipped with tools to perform specific actions (e.g., booking appointments).
    """
    def __init__(self, llm_model: ChatOpenAI):
        """
        Initializes the FitnessAgent.

        Args:
            llm_model (ChatOpenAI): The language model that the agent will use to 
                                    process queries and generate responses.
        """
        self.llm_model = llm_model
        
        system_prompt = (
            "You are a specialized agent to set, cancel or reschedule an appointment based on the query. "
            "You have access to the tool.\n Make sure to ask user politely if you need any further information to execute "
            "the tool.\n For your information, Always consider the current year is 2025."
        )
        
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("placeholder", "{messages}")]
        )
        
        # Tools are temporarily disabled to make the app runnable.
        # You will need to import and add your tools here.
        tools = []
        
        self.agent = create_react_agent(model=self.llm_model, tools=tools, messages_modifier=prompt)

    def run(self, state: State) -> dict:
        """
        Executes the main logic of the FitnessAgent.

        This method is called when the agent is activated within the LangGraph workflow.
        It processes the current state, which includes the conversation history, and 
        returns a dictionary with the agent's response.

        Args:
            state (State): The current state of the conversation graph.

        Returns:
            dict: A dictionary containing the AIMessage with the agent's response,
                  which will be added to the conversation state.
        """
        print("---EXECUTING FITNESS AGENT---")
        result = self.agent.invoke(state)
        
        # Update the state with the AIMessage from this agent's execution
        return {
            "messages": [
                AIMessage(content=result["messages"][-1].content, name="fitness_node")
            ]
        }
