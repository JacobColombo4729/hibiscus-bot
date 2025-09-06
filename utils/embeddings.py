"""
This module handles the Retrieval-Augmented Generation (RAG) implementation for the Hibiscus Bot.

It includes functions for finding and processing files (text, HTML, PDF), chunking them into
manageable pieces, ingesting them into a Chroma vector database, and retrieving relevant
chunks based on a user's query. This allows the bot to access a knowledge base of documents
to provide more accurate and context-aware responses.
"""
import glob
import chromadb
import ollama
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the ChromaDB client and create/get collections for various wellness topics.
client = chromadb.PersistentClient(path="./chroma_db")

meal_planning_collection = client.get_or_create_collection("meal_planning")
fitness_coach_collection = client.get_or_create_collection("fitness_coach")
mental_wellness_collection = client.get_or_create_collection("mental_wellness")
sleep_coach_collection = client.get_or_create_collection("sleep_coach")
workplace_ergonomics_collection = client.get_or_create_collection("workplace_ergonomics")
care_navigator_collection = client.get_or_create_collection("care_navigator")
hydration_coach_collection = client.get_or_create_collection("hydration_coach")
wellness_analytics_collection = client.get_or_create_collection("wellness_analytics")

def get_files(path, exts=('.txt', '.html', '.pdf')):
    """
    Recursively finds all files with specified extensions in a given directory.

    Args:
        path (str): The directory to search.
        exts (tuple, optional): A tuple of file extensions to look for.
                                Defaults to ('.txt', '.html', '.pdf').

    Returns:
        list: A list of file paths that match the specified extensions.
    """
    special_files = []
    for ext in exts:
        # This means search source and all subdirectories for the given extension
        special_files += glob.glob(f"{path}/**/*{ext}", recursive=True)
    return special_files

# Works for txt, html, pdf
def chunk_file(filepath, ext):
    """
    Reads a file and splits its content into smaller, manageable chunks.

    This function supports text, HTML, and PDF files. The chunking is done based on
    a fixed number of lines or at empty lines to maintain semantic coherence.

    Args:
        filepath (str): The path to the file to be chunked.
        ext (str): The extension of the file (e.g., '.pdf', '.txt').

    Returns:
        list: A list of text chunks.
    """
    if ext == '.pdf':
        reader = PdfReader(filepath)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"
        lines = full_text.splitlines()
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    # Chunk container
    chunks = []
    # Aggregators
    chunk, count = "", 0
    for line in lines:
        chunk += line
        count += 1
        # Triggers for chunking, 20 lines or empty line
        if count >= 20 or line.strip() == "":
            # If not empty line, add to the chunk
            if chunk.strip():
                chunks.append(chunk.strip())
            # Reset chunk and count
            chunk, count = "", 0
    # For remainder, add to chunks
    if chunk.strip():
        chunks.append(chunk.strip())
    return chunks

def ingest_corpus(dir, collection):
    """
    Processes a directory of documents and ingests them into a ChromaDB collection.

    This function finds all supported files in the directory, chunks them, generates
    embeddings for each chunk using OpenAI's embedding model, and then adds the
    chunks and their embeddings to the specified ChromaDB collection.

    Args:
        dir (str): The directory containing the corpus of documents.
        collection: The ChromaDB collection object to ingest the documents into.
    """
    files = get_files(dir)
    docs, ids = [], []
    # Chunks each file in the corpus and adds each chunk to docs
    # docs is the list chunks for the entire corpus
    for file in files:
        _, ext = os.path.splitext(file)
        chunks = chunk_file(file, ext)
        for i, chunk in enumerate(chunks):
            docs.append(chunk)
            ids.append(f"{os.path.basename(file)}_{i}")
    # Ingests docs returns the list of embeddings for the entire corpus, (List[List[float]]) List[float] is a single embedding
    embeddings = OpenAIEmbeddings().embed_documents(docs)
    # embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", region_name="us-east-1").embed_documents(docs)
    # this makes a Chroma collection
    collection.add(documents=docs, ids=ids, embeddings=embeddings)

def retrieve_relevant_chunks(query, collection, k):
    """
    Retrieves the most relevant document chunks from a ChromaDB collection for a given query.

    This function generates an embedding for the user's query, searches the specified
    ChromaDB collection for the top-k most similar chunks, and returns them.

    Args:
        query (str): The user's query.
        collection: The ChromaDB collection to search.
        k (int): The number of relevant chunks to retrieve.

    Returns:
        list: A list of tuples, where each tuple contains a relevant chunk of text
              and its corresponding ID.
    """
    embedding = OpenAIEmbeddings().embed_query(query)
    # embedding = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", region_name="us-east-1").embed_query(query)
    # searches chromadb for the top k most relevant items in collection
    results = collection.query(query_embeddings=[embedding], n_results=k)
    # returns a list of tuples, each tuple contains a chunk and its id
    return list(zip(results['documents'][0], results['ids'][0]))
 


# Main function for embedding docs
if __name__ == "__main__":
    """
    This block of code is executed when the script is run directly.
    It's used for ingesting a corpus of documents into the ChromaDB.
    You can specify which directories to process and which collections to use.
    """

    corpus_map = {
        "docs/MealPlanningCorpus": meal_planning_collection,
    }

    print("--- Starting Corpus Ingestion ---")

    for doc_dir, collection in corpus_map.items():
        if os.path.isdir(doc_dir):
            print(f"Processing documents in '{doc_dir}' for collection '{collection.name}'...")
            try:
                ingest_corpus(doc_dir, collection)
                print(f"Successfully processed and ingested documents from '{doc_dir}'.")
            except Exception as e:
                print(f"An error occurred while processing '{doc_dir}': {e}")
        else:
            print(f"Warning: Directory '{doc_dir}' not found. Skipping.")
    
    print("--- Corpus Ingestion Complete ---")




