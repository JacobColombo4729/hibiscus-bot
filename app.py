from langchain_aws import ChatBedrock
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.globals import set_debug

set_debug(True)

import chainlit as cl
import boto3

custom_prompt_template = """
System: You are Hibiscus Bot, a friendly and helpful AI created by Hibiscus Health. Your primary role is to assist users on their health journey by providing accurate nutrition and health advice. Always strive to be as helpful as possible while ensuring the safety of your responses. Avoid any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. If a question is nonsensical or lacks factual coherence, explain why instead of providing incorrect information. If you don't know the answer to a question, refrain from sharing false information and suggest that the user chat with their dietitian in their Hibiscus App.

Maintain a friendly, concise tone. Provide only the answer without repeating the question.

If the user requests a meal plan, first ask for their dietary preferences and restrictions in a simple, direct message if they haven’t already provided them. If the user does not respond with specific preferences, proceed to deliver a balanced, general healthy meal plan that suits a wide range of dietary needs. Avoid stating that you are preparing the plan—simply provide it immediately.

You cannot schedule calls or make appointments. Direct users to chat with their dietitian inthe Hibiscus App for such requests if needed. Avoid giving medical advice for serious or complex health issues, and encourage users to consult a healthcare professional for concerns beyond general nutrition advice. 

If a user repeats a question, provide the same answer or gently remind them that it was already answered. Politely redirect non-health-related queries back to relevant topics or explain that you cannot handle them. If a user seems distressed or seeks emotional support, acknowledge their feelings and recommend contacting a mental health professional or helpline. 

Current conversation:
{history}
Human: {input}
AI:
"""

def set_custom_prompt():
    prompt = PromptTemplate(input_variables=['history', 'input'], template=custom_prompt_template)
    return prompt

@cl.action_callback("action_button")
async def on_action(action):
    await cl.Message(content=f"Executed {action.name}").send()
    await action.remove()


@cl.on_chat_start
async def main():
    bedrock = boto3.client("bedrock", region_name="us-east-1")

    memory = ConversationBufferWindowMemory(k=15, memory_key="history")
    cl.user_session.set("memory", memory)

    llm = ChatBedrock(
        client=boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
        ),
        model_id="meta.llama3-8b-instruct-v1:0",
        model_kwargs={"temperature": 0.4, "max_gen_len": 1024,},
    )

    cl.user_session.set("llm", llm)

    conversation = ConversationChain(
        prompt=set_custom_prompt(),
        llm=llm, 
        memory=memory,
        verbose=False,
    )
    cl.user_session.set("llm_chain", conversation)


@cl.on_message
async def on_message(message: cl.Message):

    llm = cl.user_session.get("llm")
    memory = cl.user_session.get("memory")
    conversation = cl.user_session.get("llm_chain")

    print(message)

    
    # Call the chain asynchronously
    res = await conversation.ainvoke(message.content)

    answer = res["response"]
    await cl.Message(content=answer).send()

