from openai import OpenAI

def test_connection():
    api_base = "http://localhost:11434/v1"
    try:
        # We use a dummy API key because Ollama doesn't require one
        client = OpenAI(base_url=api_base, api_key="test-key")
        
        print(f"Testing connection to Ollama at {api_base}...")
        
        # Send a basic prompt
        response = client.chat.completions.create(
            model="qwen2.5", # Ensure this matches the exact tag you use in Ollama
            messages=[{"role": "user", "content": "Reply exactly with 'Connection Successful!' if you can read this."}],
            max_tokens=20
        )
        
        print("\n--- RESPONSE RECEIVED ---")
        print(response.choices[0].message.content)
        print("-------------------------")
        print("Test Passed: Your LLM backend is working perfectly!")
        
    except Exception as e:
        print("\n--- CONNECTION FAILED ---")
        print(str(e))

if __name__ == "__main__":
    test_connection()