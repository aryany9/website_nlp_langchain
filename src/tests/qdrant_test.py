from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os


# Load environment variables from .env file
load_dotenv()

# Connect to Qdrant
qdrant_client = QdrantClient(
    host=os.getenv("QDRANT_URL"),
    https=True,
    api_key=os.getenv("QDRANT_API_KEY")
)

def test_qdrant_connection():
    """
    Test the connection to Qdrant.
    """
    try:
        # Attempt to get collections to check the connection
        collections = qdrant_client.get_collections()
        assert collections is not None, "Failed to connect to Qdrant."
        print("Qdrant connection test passed.")
    except Exception as e:
        print(f"Qdrant connection test failed: {str(e)}")

def test_create_collection():
    """
    Test the creation of a collection in Qdrant.
    """
    collection_name = "test_collection"
    try:
        # Attempt to create a collection
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config={"size": 128, "distance": "Cosine"}
        )
        print(f"Collection '{collection_name}' created successfully.")
    except Exception as e:
        print(f"Failed to create collection '{collection_name}': {str(e)}")

if __name__ == "__main__":
    test_qdrant_connection()
    test_create_collection()