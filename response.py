from embeddings import retrieve_relevant_chunks
from langchain_aws import ChatBedrock

import boto3

def answer_with_rag(collection, query, k=6):

    llm = ChatBedrock(
        client=boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
        ),
        model_id="meta.llama3-8b-instruct-v1:0",
        model_kwargs={"temperature": 0.4, "max_gen_len": 1024,},
    )

    docs = retrieve_relevant_chunks(query, collection, k)  # [(text, id, score), ...] ideally

    context_blocks = "\n\n".join(
        f"[{doc_id}]\n{txt}" for (txt, doc_id, *_) in docs
    )
    prompt = f"""You are a helpful assistant. Use only the context.
    If the answer isn't in the context, say you don't know.

    CONTEXT
    {context_blocks}

    QUESTION
    {query}

    INSTRUCTIONS
    - Cite doc ids if you rely on them.
    - Be concise and factual.
    """

    return llm.invoke(prompt)

