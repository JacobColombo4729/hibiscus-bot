from langchain_aws import ChatBedrock
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.globals import set_debug
# from agent import get_agent
from embeddings import retrieve_relevant_chunks, meal_planning_collection
from prompts import supervisor_prompt_template

set_debug(True)

import chainlit as cl
import boto3

def set_custom_prompt():
    prompt = PromptTemplate(input_variables=['history', 'input'], template=supervisor_prompt_template)
    return prompt

@cl.action_callback("action_button")
async def on_action(action):
    await cl.Message(content=f"Executed {action.name}").send()
    await action.remove()


@cl.on_chat_start
async def main():

    llm = ChatBedrock(
        client=boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
        ),
        model_id="meta.llama3-8b-instruct-v1:0",
        model_kwargs={"temperature": 0.4, "max_gen_len": 1024,},
    )

    memory = ConversationBufferWindowMemory(k=15)

    conversation = ConversationChain(
        prompt=set_custom_prompt(),
        llm=llm,
        memory=memory,
        verbose=False,
    )
    cl.user_session.set("llm_chain", conversation)


@cl.on_message
async def on_message(message: cl.Message):

    # llm = cl.user_session.get("llm")
    # memory = cl.user_session.get("memory")
    conversation = cl.user_session.get("llm_chain")

    print(message)

    k_chunks = retrieve_relevant_chunks(message.content, meal_planning_collection, 6)
    print("k_chunks: ", k_chunks)
    
    # Call the chain asynchronously
    res = await conversation.ainvoke(f"Here is the user's question: {message.content}\n\nHere are the relevant chunks: {k_chunks}, Respond in a friendly and helpful manner. If you don't know the answer, say you don't know. Do not hallucinate.")

    answer = res["response"]
    await cl.Message(content=answer).send()

