import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from transformers import pipeline

# Load environment variables
load_dotenv()

# Initialize Pinecone client
index_name = "langchain-demo"
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Use a smaller model for CPU inference (e.g., DistilGPT-2)
generator = pipeline("text-generation", model="distilgpt2", device=-1)  # Use device=-1 for CPU

def search_pinecone(query, index_name="langchain-demo"):
    """Search Pinecone for the most relevant document to the query."""
    index = pc.Index(index_name)

    # Reframe query for better embedding results
    enhanced_query = f"Find relevant information for: {query}"
    
    embeddings = HuggingFaceEmbeddings()
    query_embedding = embeddings.embed_query(enhanced_query)

    # Perform search
    search_results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )

    # Extract relevant text
    retrieved_texts = [result['metadata']['text'] for result in search_results['matches']]
    return " ".join(retrieved_texts) if retrieved_texts else "No relevant information found."

def generate_response_with_generator(retrieved_text, user_input):
    """Generate a refined response using a smaller model (e.g., DistilGPT-2) with enhanced prompt engineering."""
    
    # Enhanced prompt
    prompt = f"""
    You are an AI assistant providing details about the Bangladesh House Building Finance Corporation.

    User Query: "{user_input}"
    Retrieved Information: "{retrieved_text}"

    Based on the retrieved information, provide a brief yet informative response:
    """

    # Generate response
    response = generator(prompt, max_length=153, num_return_sequences=1, do_sample=True)

    # Extract and return text
    generated_text = response[0]['generated_text'].strip()
    return generated_text

def main():
    """Main function to handle user queries and generate responses."""
    user_input = "How many loan options are available in BHBFC?"  # Dynamic user input
    retrieved_text = search_pinecone(user_input)

    response = generate_response_with_generator(retrieved_text, user_input)
    print("\nChatbot Response:", response)

if __name__ == "__main__":
    main()
