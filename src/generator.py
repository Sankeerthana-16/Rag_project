import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def generate_answer(question, context):

    prompt = f"""
    You are a helpful AI assistant.

    Answer the question using only the provided context.
    If the answer is not available in the context,
    say "I don't know based on the provided document."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error communicating with LLM: {str(e)}"