import json
import boto3
from langchain_aws import ChatBedrock
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

def lambda_handler(event, context):
    
    
    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
    
    user_prompt = event.get("prompt", "Default prompt text if not provided.")


    # Embed the prompt in Llama 3's instruction format.
    formatted_prompt = f"""
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    You are Hibiscus Bot, a friendly helpful AI created by Hibiscus Health. Your role is to help the user with their health journey by giving nutrition and health advice. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information and ask the user to schedule a call with their dietitian.

    Maintain a friendly and helpful tone. Keep it short and
    only return your answer and nothing else. Do not repeat the question.
    
    <|start_header_id|>user<|end_header_id|>
    {user_prompt}
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
    
    prompt = {
        "prompt": formatted_prompt,
        "max_gen_len": 1024,
        "temperature": 0.8,
        "top_p": 0.9
    }
        
    input = {
     "modelId": "meta.llama3-8b-instruct-v1:0",
     "contentType": "application/json",
     "accept": "application/json",
     "body": json.dumps(prompt)
    }
    
    response = bedrock.invoke_model(
        body=input['body'], 
        modelId=input['modelId'], 
        accept=input['accept'],
        contentType=input['contentType'])
    
    results = json.loads(response['body'].read())
    print(results)
    
    return {
        'statusCode': 200,
        'body': json.dumps({'result': results})
    }

