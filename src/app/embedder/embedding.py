
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def get_embedding_model():
    """
    Get the Google Generative AI Embeddings model.
    :return: GoogleGenerativeAIEmbeddings instance
    """
    import os
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()

    # Initialize the Google Generative AI Embeddings model
    embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=os.getenv("GEMINI_API_KEY"))

    return embedding

