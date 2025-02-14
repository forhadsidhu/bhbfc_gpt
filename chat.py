import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from transformers import pipeline
import numpy as np

# Load environment variables
load_dotenv()


# Initialize Pinecone client and index
index_name = "langchain-demo"
pc = Pinecone(
api_key=os.environ.get("PINECONE_API_KEY")  # Ensure your API key is loaded correctly
)



def search_pinecone(query, index_name="langchain-demo"):
    """Search Pinecone for the most relevant document to the query"""
    
    index = pc.Index(index_name)

    # Embed the query using Hugging Face embeddings
    embeddings = HuggingFaceEmbeddings()
    query_embedding = embeddings.embed_query(query)  # Use embed_query instead of embed

    # Perform the vector search in Pinecone
    search_results = index.query(
        vector=query_embedding,
        top_k=3,  # Number of similar documents to retrieve
        include_metadata=True
    )

    # Extract the most relevant document
    retrieved_text = " ".join([result['metadata']['text'] for result in search_results['matches']])
    
    return retrieved_text


def generate_response_with_huggingface(retrieved_text, user_input):
    """Generate a response using a Hugging Face model with context from Pinecone query"""

    # Load the model pipeline (you can change to a different model if needed)
    generator = pipeline("text-generation", model="gpt2")  # You can use GPT-2 or any other suitable model
    
    # Combine the user input and retrieved information to form the prompt
    prompt = f"User asked: {user_input}\n\nThe relevant information is: {retrieved_text}\n\nProvide a two line, human-like response based on retrieved text."
    
    # Generate the response
    response = generator(prompt, max_length=560, num_return_sequences=1)

    # Extract the text from the response
    generated_text = response[0]['generated_text'].strip()


    two_line_response = generated_text.split("\n")[-2:]

    return two_line_response


def main():
    """Main function to call embedding creation and chatbot response generation"""
    
    # Example: Get user input and retrieve relevant text from Pinecone
    user_input = "Will i get jobs that fit me?"

    # Retrieve relevant text from Pinecone based on the user's query
    retrieved_text = search_pinecone(user_input)

    # Step 2: Generate a response using the retrieved text and user input
    response = generate_response_with_huggingface(retrieved_text, user_input)
    print("Chatbot Response:", response)


if __name__ == "__main__":
    main()