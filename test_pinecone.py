import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import numpy as np 
load_dotenv()



# Initialize Pinecone with the API key
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")  # Ensure your API key is loaded correctly
)

# Now you can interact with the Pinecone index
# Example: Querying Pinecone after index creation

# Define index name (ensure this matches your created index)
index_name = "langchain-demo"

# Access the index
index = pc.Index(index_name)

# You can now perform actions such as querying the index, inserting vectors, etc.
# For example, to perform a vector search:
query_vector = np.random.rand(768).tolist()  # Create a random vector with dimension 768 (for testing purposes)

# Now, perform the query with the correct vector dimension
results = index.query(
    vector=query_vector,
    top_k=3,  # Number of most similar results to fetch
    include_metadata=True  # Optionally include metadata
)

# Display search results
print(results)
