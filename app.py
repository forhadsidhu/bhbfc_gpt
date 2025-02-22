import os
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from llama_cpp import Llama

# 🔹 Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INEX_NAME = os.getenv("")

if not PINECONE_API_KEY:
    st.error("🔴 Missing Pinecone API key! Check your .env file.")
    st.stop()

# 🔹 Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = PINECONE_INEX_NAME
index = pc.Index(INDEX_NAME)

# 🔹 Use Local Embedding Model (Sentence Transformers)
# Use a model that outputs 768-dimensional embeddings
embedding_model = SentenceTransformer("all-mpnet-base-v2")  # ✅ Fixes the issue

# 🔹 Load Llama model
llm = Llama(
    model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=6,
    n_batch=64
)

# 🔹 Streamlit UI setup
st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="centered")
st.title("📚 AI Chatbot with Pinecone RAG")

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask me anything...")

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 🔹 Retrieve relevant documents from Pinecone
    try:
        query_embedding = embedding_model.encode(user_input).tolist()
        search_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)

        # Extract retrieved context
        retrieved_texts = [match["metadata"].get("text", "") for match in search_results.get("matches", [])]
        context = "\n".join(retrieved_texts) if retrieved_texts else "No relevant documents found."

    except Exception as e:
        st.error(f"🔴 Error querying Pinecone: {e}")
        context = "No relevant documents found."

    # 🔹 Generate response with context
    with st.chat_message("assistant"):
        try:
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "Use the retrieved context to answer accurately."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_input}"}
                ]
            )
            bot_reply = response["choices"][0]["message"]["content"]

        except Exception as e:
            st.error(f"🔴 Error generating response: {e}")
            bot_reply = "I'm sorry, I couldn't generate a response."

        st.markdown(bot_reply)

    # Store AI response
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
