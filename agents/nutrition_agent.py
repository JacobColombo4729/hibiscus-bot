"""
This module defines the NutritionAgent, a specialized agent focused on providing 
nutrition-related information and answering frequently asked questions.

The NutritionAgent is designed to be part of a larger agentic system, where it can be 
invoked by a supervisor agent to handle specific user queries about nutrition and diet.
"""
from typing import Literal
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from .state import State



class NutritionAgent:
    """
    A specialized agent that handles nutrition-related queries.

    This agent is initialized with a language model and is designed to work within
    a LangGraph workflow. It uses a predefined system prompt to guide its behavior
    and can be equipped with tools to perform specific actions (e.g., fetching data).
    """
    def __init__(self, llm_model: ChatOllama):
        """
        Initializes the NutritionAgent.

        Args:
            llm_model (ChatOpenAI): The language model that the agent will use to 
                                    process queries and generate responses.
        """
        self.llm_model = llm_model
        
        system_prompt = (
            "You are a specialized agent to provide information related to availability of doctors "
            "or any FAQs related to hospital based on the query. You have access to the tool.\n Make sure to ask "
            "user politely if you need any further information to execute the tool.\n For your information, Always "
            "consider current year is 2025."
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
        Executes the main logic of the NutritionAgent.

        This method is called when the agent is activated within the LangGraph workflow.
        It processes the current state, which includes the conversation history, and 
        returns a dictionary with the agent's response.

        Args:
            state (State): The current state of the conversation graph.

        Returns:
            dict: A dictionary containing the AIMessage with the agent's response,
                  which will be added to the conversation state.
        """
        print("---EXECUTING NUTRITION AGENT---")
        result = self.agent.invoke(state)
        
        # Update the state with the AIMessage from this agent's execution
        # and tell the graph to return to the supervisor.
        return {
            "messages": [
                AIMessage(content=result["messages"][-1].content, name="nutrition_node")
            ]
        }
