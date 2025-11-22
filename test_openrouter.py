# test_openrouter.py
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
print(f"API Key: {api_key[:20] if api_key else 'NOT FOUND'}...")

try:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )
    
    response = client.chat.completions.create(
        model="anthropic/claude-3-haiku",
        messages=[{"role": "user", "content": "Say 'test successful'"}],
        max_tokens=50
    )
    
    print("✅ Success!")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"❌ Error: {e}")