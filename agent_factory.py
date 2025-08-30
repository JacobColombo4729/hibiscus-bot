from typing import Literal, List, Any
from langchain_core.tools import tool
from langgraph.types import Command
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
from langchain_core.prompts.chat import ChatPromptTemplate
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain.agents import Tool
from langchain_core.messages import HumanMessage, AIMessage
import boto3
from langchain_aws import ChatBedrock
# from prompt_library.prompt import system_prompt
# from toolkit.toolkits import *

class Router(TypedDict):
    next: Literal["information_node", "booking_node", "FINISH"]
    reasoning: str

class AgentState(TypedDict):
    messages: Annotated[list[Any], add_messages]
    id_number: int
    next: str
    query: str
    current_reasoning: str

""" I want this agent to generate diet plans for the following medical conditions:
Hypertension (high blood pressure), High cholesterol, Heart failure, Diabetes,
Obesity, Hyperthyroidism, Hypothyroidism, Chronic kidney disease, Liver cirrhosis,
Gout, Celiac disease, Irritable bowel syndrome (IBS), GERD (acid reflux), Lactose intolerance,
Osteoporosis, Arthritis, Food allergies, Inflammatory bowel disease (IBD) (e.g., Crohn's disease, ulcerative colitis)
"""
class MedConditionDietingAgent:
    def __init__(self):
        llm_model = ChatBedrock(
                        client=boto3.client("bedrock-runtime", region_name="us-east-1"),
                        model_id="meta.llama3-8b-instruct-v1:0",
                        model_kwargs={"temperature": 0.4, "max_gen_len": 1024,},
                    )
        self.llm_model=llm_model.get_model()
        self.rag_tool = Tool(
                            name="KnowledgeBaseSearch",
                            func=lambda query: rag_tool_func(query, retriever),
                            description="Use this tool to answer general questions about Hibiscus Health"
                        )
        self.docs_path = "docs/MedConditionDieting"

    def get_agent(username):
        session = cl.user_session
        # if not session.get("retriever"):
        #     # Update load_support_docs to actually load documents
        #     support_docs = load_support_docs()
        #     session.set("retriever", build_retriever(support_docs))
        if not session.get("user_agents"):
            session.set("user_agents", {})
        user_agents = session.get("user_agents")
        if username not in user_agents:
            user_agents[username] = MedConditionDietingAgent(retriever=session.get("retriever"))
            session.set("user_agents", user_agents)
        return user_agents[username]

    def supervisor_node(self, state: AgentState) -> Command[Literal['information_node', 'mealplanning_node', '__end__']]:
        print("**************************below is my state right after entering****************************")
        print(state)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"user's identification number is {state['id_number']}"},
        ] + state["messages"]

        print("***********************this is my message*****************************************")
        print(messages)

        # query = state['messages'][-1].content if state["messages"] else ""
        query = ''
        if len(state['messages']) == 1:
            query = state['messages'][0].content

        print("************below is my query********************")
        print(query)

        llm_response = self.llm_model.invoke(messages)
        response = {"next": "information_node", "reasoning": "Hardcoded to information_node for testing"}
        goto = response["next"]

        print("********************************this is my goto*************************")
        print(goto)

        print("********************************")
        print(response["reasoning"])

        if goto == "FINISH":
            goto = END

        print("**************************below is my state****************************")
        print(state)

        if query:
            return Command(goto=goto, update={'next': goto,
                                              'query': query,
                                              'current_reasoning': response["reasoning"],
                                              'messages': [HumanMessage(content=f"user's identification number is {state['id_number']}")]
                            })
        return Command(goto=goto, update={'next': goto, 'current_reasoning': response["reasoning"]})

    def information_node(self, state: AgentState) -> Command[Literal['supervisor']]:
        print("*****************called information node************")

        # I want to pull data on the medical condition straight from user data.
        # If you are asked a question about Hibiscus Health, nutrition, or anything in the knowledge base, always use the KnowledgeBaseSearch tool to find the answer.
        system_prompt = "You are a specialized agent designed to provide information on dietary guidance for the medical condition identified in user data. You have access to the tool.\n Make sure to ask user politely if you need any further information to execute the tool."

        system_prompt = ChatPromptTemplate.from_messages(
                [("system", system_prompt), ("placeholder", "{messages}")]
        )

        information_agent = create_react_agent(model=self.llm_model,tools=[check_availability_by_doctor,check_availability_by_specialization] ,prompt=system_prompt)

        result = information_agent.invoke(state)

        return Command(
            update={
                "messages": state["messages"] + [
                    AIMessage(content=result["messages"][-1].content, name="information_node")
                    #HumanMessage(content=result["messages"][-1].content, name="information_node")
                ]
            },
            goto="supervisor",
        )

    def mealplanning_node(self, state: AgentState) -> Command[Literal['supervisor']]:
        print("*****************called booking node************")

        system_prompt = "You are specialized agent to set, cancel or reschedule appointment based on the query. You have access to the tool.\n Make sure to ask user politely if you need any further information to execute the tool.\n For your information, Always consider current year is 2024."

        system_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        system_prompt
                    ),
                    (
                        "placeholder",
                        "{messages}"
                    ),
                ]
            )
        booking_agent = create_react_agent(model=self.llm_model,tools=[set_appointment,cancel_appointment,reschedule_appointment],prompt=system_prompt)

        result = booking_agent.invoke(state)

        return Command(
            update={
                "messages": state["messages"] + [
                    AIMessage(content=result["messages"][-1].content, name="booking_node")
                    #HumanMessage(content=result["messages"][-1].content, name="booking_node")
                ]
            },
            goto="supervisor",
        )

    def workflow(self):
        self.graph = StateGraph(AgentState)
        self.graph.add_node("supervisor", self.supervisor_node)
        self.graph.add_node("information_node", self.information_node)
        self.graph.add_node("booking_node", self.booking_node)
        self.graph.add_edge(START, "supervisor")
        self.app = self.graph.compile()
        return self.app








