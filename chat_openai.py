import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
import openai  # OpenAI API

# Load environment variables
load_dotenv()

# Initialize Pinecone client
index_name = "langchain-demo"
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# OpenAI API Key
openai.api_key = os.environ.get("OPENAI_API_KEY")

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

def generate_response_with_openai(retrieved_text, user_input):
    """Generate a refined response using OpenAI GPT."""
    
    # Enhanced few-shot prompt
    prompt = f"""
    You are an AI assistant providing information about Bangladesh House Building Finance Corporation (BHBFC).

    User Query: "{user_input}"
    Retrieved Information: "{retrieved_text}"

    Based on the retrieved information, provide a concise and informative response:
    """

    # OpenAI GPT API call
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # You can use "gpt-4" for better accuracy
        messages=[{"role": "system", "content": "You are a helpful AI assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=150
    )

    # Extract response text
    return response["choices"][0]["message"]["content"].strip()

def main():
    """Main function to handle user queries and generate responses."""
    user_input = "How many loan options are available in BHBFC?"  # Dynamic user input
    retrieved_text = search_pinecone(user_input)
    print("\nRetrieved Text:", retrieved_text)

    response = generate_response_with_openai(retrieved_text, user_input)
    print("\nChatbot Response:", response)

if __name__ == "__main__":
    main()
