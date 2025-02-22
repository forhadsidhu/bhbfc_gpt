from llama_cpp import Llama

# Load the GGUF model from a local path
llm = Llama(
    model_path="C:\\Users\\User\\.ollama\\models\\Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",  # Adjust the path if necessary
    n_ctx=4096  # Set context length (adjust based on hardware)
)

# Generate a response
response = llm.create_chat_completion(
    messages=[
        {"role": "user", "content": "What is the Bangladesh housebuilding finance corporation?"}
    ]
)

print(response["choices"][0]["message"]["content"])  # Extract response text
