import os
import re
import requests
# import fitz  # PyMuPDF for PDF processing
from requests.exceptions import ConnectionError, Timeout, RequestException
from langchain.schema import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws.embeddings import BedrockEmbeddings
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set USER_AGENT environment variable with a fallback
USER_AGENT = os.getenv('USER_AGENT')

# Function to check if a document exists in the vectorstore
def document_exists(vectorstore, doc, embedding_function):
    query_text = doc.page_content[:500]  # Using the first 500 characters
    query_embedding = embedding_function.embed_query(query_text)
    results = vectorstore.similarity_search_by_vector(query_embedding, k=1)
    if results:
        for result in results:
            if result.metadata == doc.metadata:
                return True
    return False

# URL validation function
def is_valid_url(url):
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # ...or ipv4
        r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # ...or ipv6
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, url) is not None

# Load URLs from a file
def load_urls_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            urls = file.readlines()
        # Strip whitespace and newlines
        urls = [url.strip() for url in urls]
        return urls
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []

# Updated TimeoutWebBaseLoader to include User-Agent header
class TimeoutWebBaseLoader(WebBaseLoader):
    def __init__(self, web_path, timeout=10):
        super().__init__(web_path)
        self.timeout = timeout
        self.headers = {
            "User-Agent": os.getenv('USER_AGENT', 'nih_diabetic_smoothie/1.0 (jacob@hibiscushealth.com)')
        }

    def scrape(self):
        try:
            response = requests.get(self.web_path, timeout=self.timeout, headers=self.headers)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            return response.text
        except (RequestException, Timeout) as e:
            print(f"Error loading {self.web_path}: {str(e)}")
            return None

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    try:
        pdf_document = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        pdf_document.close()
        if text:
            # Print a snippet of the text
            print(f"Extracted text from {pdf_path}: {text[:200]}...")
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {str(e)}")
        return None

# Function to load PDFs from a folder with directory existence check
def load_pdfs_from_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"Directory not found: {folder_path}")
        return []

    pdf_texts = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            text = extract_text_from_pdf(pdf_path)
            if text:
                pdf_texts.append(text)
    return pdf_texts

# Preprocess text to handle long continuous strings
def preprocess_text(text):
    # Insert artificial breaks for long continuous strings
    max_len = 500  # Break up long continuous strings after 500 characters
    processed_text = re.sub(r'(\S{500,})', lambda m: ' '.join([m.group(0)[i:i + max_len] for i in range(0, len(m.group(0)), max_len)]), text)
    return processed_text

# Function to convert text to documents with tags
def convert_text_to_documents(text, source="unknown", tags=[]):
    # Preprocess text to handle long continuous strings
    text = preprocess_text(text)
    # Join the tags list into a comma-separated string
    tags_str = ",".join(tags)
    return [Document(page_content=text, metadata={"source": source, "tags": tags_str})]

# Function to process URLs and store embeddings with dynamic tags
def process_urls_and_store_embeddings(file_path, persist_directory, tags=[]):
    urls = load_urls_from_file(file_path)
    valid_urls = [url for url in urls if is_valid_url(url)]
    if not valid_urls:
        print("No valid URLs provided.")
        return

    docs_list = []
    for url in valid_urls:
        try:
            loader = TimeoutWebBaseLoader(url, timeout=30)  # 30 seconds timeout
            docs = loader.scrape()
            if docs:
                TODO
                # Merge user-provided tags with URL-specific tags
                url_tags = tags.copy()  # Create a copy of tags
                if "diabetes" in url or "obesity" in url:
                    url_tags.append("obesity")
                    url_tags.append("type2diabetes")

                docs_list.extend(convert_text_to_documents(docs, source=url, tags=url_tags))
            else:
                print(f"No content loaded from {url}")
        except (ConnectionError, Timeout) as e:
            print(f"Timeout or connection error with {url}: {str(e)}")
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")

    if not docs_list:
        print("No documents were successfully loaded.")
        return

    store_documents(docs_list, persist_directory)

# Function to process CSV and store embeddings with dynamic tags
def process_csv_and_store_embeddings(csv_file_path, persist_directory, tags=[]):
    try:
        df = pd.read_csv(csv_file_path)
    except pd.errors.ParserError:
        print("Error encountered with default CSV reading. Attempting alternative methods...")
        try:
            df = pd.read_csv(csv_file_path, sep='\t')  # Try tab as delimiter
        except pd.errors.ParserError:
            print("Tab delimiter didn't work. Trying to infer CSV dialect...")
            try:
                with open(csv_file_path, 'r', newline='') as csvfile:
                    dialect = csv.Sniffer().sniff(csvfile.read(1024))
                    csvfile.seek(0)
                    df = pd.read_csv(csv_file_path, dialect=dialect)
            except Exception as e:
                print(f"All attempts to read CSV failed. Error: {str(e)}")
                print("Please check your CSV file for inconsistencies.")
                return

    print(f"Successfully read CSV file. Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    df['combined_text'] = df.apply(lambda row: "\n".join([
        f"Column {i}: {value}" for i, value in enumerate(row) if pd.notna(value)
    ]), axis=1)

    # Convert DataFrame rows to documents
    docs_list = [
        Document(
            page_content=row['combined_text'],
            metadata={
                "source": csv_file_path,
                "tags": ",".join(tags + ["csv_import"]),
                "row_index": index  # Use row index as a unique identifier
            }
        ) for index, row in df.iterrows()
    ]

    store_documents(docs_list, persist_directory)

# Function to process PDFs and store embeddings with dynamic tags
def process_pdfs_and_store_embeddings(folder_path, persist_directory, tags=[]):
    pdf_texts = load_pdfs_from_folder(folder_path)
    if not pdf_texts:
        print("No PDFs found or no text extracted.")
        return

    docs_list = []
    for text in pdf_texts:
        pdf_tags = tags.copy()  # Avoid modifying the original tags list
        docs = convert_text_to_documents(text, tags=pdf_tags)
        docs_list.extend(docs)

    if not docs_list:
        print("No documents were successfully loaded.")
        return

    store_documents(docs_list, persist_directory)

# Function to store documents in the vectorstore
def store_documents(docs_list, persist_directory):
    # Use a RecursiveCharacterTextSplitter to handle edge cases more robustly
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=300
    )

    doc_splits = text_splitter.split_documents(docs_list)

    # Ensure the model is pulled before using it
    embedding_function = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        region_name="us-east-1"
    )

    if os.path.exists(persist_directory):
        # Load the existing vectorstore if the directory exists
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
            # collection_name="smoothiescollection"
        )
    else:
        # Create a new vectorstore if it does not exist
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            embedding=embedding_function,
            persist_directory=persist_directory,
            # collection_name="smoothiescollection"
        )
        print("New Chroma collection created and documents added.")
        return

    new_docs = []
    for doc in doc_splits:
        if not document_exists(vectorstore, doc, embedding_function):
            new_docs.append(doc)

    if new_docs:
        vectorstore.add_documents(new_docs)
        print(f"{len(new_docs)} new documents added to the existing collection.")
    else:
        print("No new documents to store.")

    # Chroma automatically persists the data to disk as long as `persist_directory` is provided
    print("Chroma collection updated and persisted successfully.")

# Main function to run both URL and PDF processing with tags
if __name__ == "__main__":
    persist_directory = "chroma_db"

    # FOR MED CONDITION AGENT:
    # Example: Process URLs with tags
    hypertension_file_path = 'docs/MedConditionDietSources/hypertension.txt'
    url_tags_hypertension = [
        "hypertension", "dash", "dash eating plan", "following dash", "living with dash"
    ]
    process_urls_and_store_embeddings(hypertension_file_path, persist_directory, tags=url_tags_hypertension)

