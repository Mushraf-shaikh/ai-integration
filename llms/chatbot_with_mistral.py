import os
import aiohttp
import asyncio
import json
from dotenv import load_dotenv

load_dotenv()


async def stream_response(prompt):
    api_key = os.getenv('MISTRAL_API_KEY')
    model = "mistral-large-latest"
    url = "https://api.mistral.ai/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": True
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            async for line in response.content:
                if line.startswith(b'data:'):
                    chunk = line[len(b'data:'):].strip()
                    if chunk:
                        try:
                            chunk_data = json.loads(chunk.decode('utf-8'))
                            if 'choices' in chunk_data and chunk_data['choices']:
                                content = chunk_data['choices'][0]['delta'].get('content', '')
                                print(content, end='', flush=True)
                        except json.JSONDecodeError as e:
                            print(f"Failed to decode chunk: {chunk}. Error: {e}")

    print("\nStreaming complete.")
    stream_chat_response()


def stream_chat_response():
    user_prompt = input("Enter your prompt: ")
    asyncio.run(stream_response(user_prompt))


if __name__ == '__main__':
    stream_chat_response()
