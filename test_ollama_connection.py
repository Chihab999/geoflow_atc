import sys

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai library not installed. Please run 'pip install openai'")
    sys.exit(1)

def test_connection():
    api_base = "http://localhost:11434/v1"
    print(f"Testing connection to OpenAI compatible endpoint at: {api_base}")
    
    try:
        # Initialize the OpenAI client for Ollama
        client = OpenAI(
            base_url=api_base,
            api_key="ollama-local" # required theoretically, but practically ignored by Ollama
        )
        
        # 1. Test Listing Models
        print("\n1. Requesting model list...")
        models = client.models.list()
        
        available_models = [m.id for m in models.data]
        print("Success! Available models on this server:")
        for m in available_models:
            print(f"  - {m}")
            
        if not available_models:
            print("Warning: No models are currently pulled in Ollama.")
            print("Run 'ollama run qwen2.5' on your server terminal to download the model.")
            sys.exit(1)
            
        # 2. Test specific completion with qwen2.5 or whatever is found
        target_model = "qwen2.5" if "qwen2.5" in available_models else available_models[0]
        
        print(f"\n2. Testing chat completion with model: {target_model}...")
        response = client.chat.completions.create(
            model=target_model,
            messages=[{"role": "user", "content": "Hello! Reply with 'Connection successful!' if you receive this."}],
            max_tokens=30
        )
        
        reply = response.choices[0].message.content.strip()
        print(f"\nSuccess! LLM responded:\n{reply}")
        print("\nThe API connection is working perfectly.")
        
    except Exception as e:
        print(f"\nCONNECTION FAILED! Details:")
        print(f"--------------------------------------------------")
        print(str(e))
        print(f"--------------------------------------------------")
        print("Troubleshooting steps:")
        print("1. Ensure Ollama is running (e.g., systemctl status ollama or just 'ollama serve')")
        print("2. Ensure OLLAMA_HOST is binding to the correct port if it's not strictly localhost.")

if __name__ == "__main__":
    test_connection()
