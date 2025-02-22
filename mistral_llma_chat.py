from llama_cpp import Llama

# Load the optimized GGUF model
llm = Llama(
    model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_ctx=2048,  # Reduce context size to avoid RAM overflow
    n_threads=6,  # Use 6 threads (keep 2 free for system stability)
    n_batch=64   # Lower batch size for better performance on limited RAM
)


# Generate a response
response = llm.create_chat_completion(
    messages=[{"role": "user", "content": "What is the Bangladesh House Building Finance Corporation?"}]
)

# Print response text
print(response["choices"][0]["message"]["content"])
