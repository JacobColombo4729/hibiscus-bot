"""
This module defines the GeneralAgent, a versatile agent that handles general wellness queries 
and conversations that do not fall into the specialized domains of nutrition or fitness.

The GeneralAgent is designed to be a "catch-all" agent in the agentic system, providing helpful 
advice and engaging in friendly conversation.
"""
from typing import Literal
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from .state import State

class GeneralAgent:
    """
    A general-purpose agent for handling non-specialized wellness queries.

    This agent is initialized with a language model and is designed to work within
    a LangGraph workflow. It uses a predefined system prompt to guide its behavior
    and does not have any specialized tools.
    """
    def __init__(self, llm_model: ChatOpenAI):
        """
        Initializes the GeneralAgent.

        Args:
            llm_model (ChatOpenAI): The language model that the agent will use to 
                                    process queries and generate responses.
        """
        self.llm_model = llm_model
        
        system_prompt = (
            "You are a general wellness assistant. Handle queries that do not fall into specific categories "
            "like nutrition or fitness. Provide helpful, general advice and engage in friendly conversation."
        )
        
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("placeholder", "{messages}")]
        )
        
        # The general agent has no specialized tools.
        tools = []
        
        self.agent = create_react_agent(model=self.llm_model, tools=tools, messages_modifier=prompt)

    def run(self, state: State) -> dict:
        """
        Executes the main logic of the GeneralAgent.

        This method is called when the agent is activated within the LangGraph workflow.
        It processes the current state, which includes the conversation history, and 
        returns a dictionary with the agent's response.

        Args:
            state (State): The current state of the conversation graph.

        Returns:
            dict: A dictionary containing the AIMessage with the agent's response,
                  which will be added to the conversation state.
        """
        print("---EXECUTING GENERAL AGENT---")
        result = self.agent.invoke(state)
        
        # Update the state with the AIMessage from this agent's execution
        return {
            "messages": [
                AIMessage(content=result["messages"][-1].content, name="general_node")
            ]
        }
