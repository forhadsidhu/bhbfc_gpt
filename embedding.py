from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from pinecone import ServerlessSpec
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

import pinecone
import os
import os
import pinecone
import numpy as np
from pinecone import Pinecone as PC
from dotenv import load_dotenv
load_dotenv()


def create_embedding():
    """Create embedding using Hugging Face embeddings"""

    loader = TextLoader('dataset/data.txt')
    documents = loader.load()

    # Split text into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
    docs = text_splitter.split_documents(documents)  

    embeddings = HuggingFaceEmbeddings()

    print("Embedding created successfully")
    return docs, embeddings  


def create_pdf_embedding():
    """Create embedding using Hugging Face embeddings from a PDF file"""

    # Load PDF document
    loader = PyMuPDFLoader('dataset/dataset_one.pdf')
    documents = loader.load()

    # Split text into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
    docs = text_splitter.split_documents(documents)

    # Initialize Hugging Face embeddings
    embeddings = HuggingFaceEmbeddings()

    print("Embedding created successfully")
    return docs, embeddings



def store_embedding(docs, embeddings):
    """Store the embedding in Pinecone vector database"""

    # Initialize Pinecone client
    pc = PC(api_key=os.getenv('PINECONE_API_KEY'))

    # Define Index Name
    index_name = "langchain-demo"

    # Checking Index
    if index_name not in pc.list_indexes():
        
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
            deletion_protection="disabled"
            )



        # Link to the new index and store embeddings
        docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
        print("Stored in new vector database successfully")

    else:
        # Link to the existing index
        docsearch = Pinecone.from_existing_index(index_name, embeddings)
        print("Stored in existing vector database successfully")





def main():
    """Main function to call embedding creation"""
  

    docs,embeddings = create_pdf_embedding()
    store_embedding(docs,embeddings)
    
if __name__ == "__main__":
    main()


