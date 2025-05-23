
from google import genai
from google.genai import types
import os


def get_answer_from_llm(prompt: str, context: str) -> str:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    """
    Get an answer from the Gemini model using the provided prompt and context.
    
    Args:
        prompt (str): The prompt to send to the model.
        context (str): The context to provide to the model.
        
    Returns:
        str: The model's response.
    """
    # Set the system prompt
    system_prompt = f"""
    You are a helpful Website Chatbot. You will act as a Agent of Website and reply like its your own.
    User will ask you a question about the website.
    You will answer the question based on the context provided.
    If the question is not related to the website, you will say 'I don't know'.
    You will always answer in a concise and clear manner.
    You will always answer in readme markdown format.
    You will also provide the document url of the information you used to answer the question.
    
    If you don't know the answer, say 'I don't know'. 
    Please be concise and clear in your response.

    Your answer should be formatted neatly and clearly, with proper punctuation and grammar.

    You'll always answer in readme markdown format.

    You will also provide the document url of the information you used to answer the question.
    Context: {context}
    """
    

    # Call the model
    response = client.models.generate_content(
                model="gemini-2.0-flash", 
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                ),
                contents=prompt
            )

    return response.text