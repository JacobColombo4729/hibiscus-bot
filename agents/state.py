"""
This module defines the data structures that represent the state of the agentic workflow.

The `State` class is a TypedDict that holds the shared state of the conversation, 
which is passed between different nodes (agents) in the LangGraph.
"""
import os
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages



# class Router(TypedDict):
#     """
#     Defines the routing decision made by the supervisor.
#
#     This class is currently commented out but can be used to structure the
#     output of the supervisor's routing logic.
#
#     Attributes:
#         next (Literal): The next node to execute in the graph.
#         reasoning (str): A brief explanation of why this route was chosen.
#     """
#     # The next node to execute in the graph.
#     next: Literal["nutrition_node", "fitness_node", "general_node", "__end__"]
#     # A brief explanation of why this route was chosen.
#     reasoning: str

class State(TypedDict):
    """
    Represents the shared state of the agentic workflow.

    This TypedDict is passed between all nodes in the graph and is used to
    maintain the conversation history and determine the next step in the workflow.

    Attributes:
        messages (Annotated[list, add_messages]): A list of messages in the conversation.
            The `add_messages` function ensures that new messages are appended to this
            list rather than overwriting it.
        next (str): A string indicating the next node to execute in the graph.
    """
    messages: Annotated[list, add_messages]
    next: str




