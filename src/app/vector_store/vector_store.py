import os
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client import QdrantClient
from termcolor import colored
from typing import List, Optional, Dict, Any
from hashlib import md5

from cli.spinner import spinner

# Connect to Qdrant
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

@spinner("Storing Chunks in Qdrant")
def create_embeddings_in_qdrant(chunks: list, embedding_model, collection_name: str):
    """
    Store chunks in Qdrant with embeddings.

    Args:
        chunks (list): List of chunks to store.
        embedding_model: The embedding model to use.
        collection_name (str): The name of the Qdrant collection.
    """
    print(colored("🔹 Storing Chunks in Qdrant  ", "yellow"))


    dummy_embedding = embedding_model.embed_query("test")
    vector_size = len(dummy_embedding)
    
    # Create collection manually if it doesn't exist
    try:
        collections = qdrant_client.get_collections().collections
        collection_names = [col.name for col in collections]

        # If the collection doesn't exist, create it
        if collection_name not in collection_names:
            qdrant_client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(colored(f"🔹 Collection '{collection_name}' created in Qdrant.", "cyan"))
        else:
            print(colored(f"🔹 Collection '{collection_name}' already exists.", "cyan"))
    except Exception as e:
        print(colored(f"❌ Error while checking/creating collection: {str(e)}", "red"))
        return

    # Create a Qdrant vector store
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embedding_model
    )

    print(colored(f"🔹 Pointing embedded chunks in '{collection_name}'.", "green"))

    # Convert your chunks to list of Documents
    documents_to_add = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk['content'],
            metadata={"source": chunk['source']}
        )
        documents_to_add.append(doc)

    # Now add all at once
    vector_store.add_documents(documents_to_add)

    print(colored(f"🔹 Stored {len(documents_to_add)} chunks in Qdrant collection '{collection_name}'.", "green"))


def compute_chunks_hash(chunks: list) -> str:
    """
    Compute a hash based on the chunks' content to detect changes.
    """
    m = md5()
    for chunk in chunks:
        m.update(chunk['content'].encode('utf-8'))
        m.update(chunk['source'].encode('utf-8'))
    return m.hexdigest()

@spinner("Syncing Chunks with Qdrant")
def sync_chunks_with_qdrant(chunks: list, embedding_model, collection_name: str):
    """
    Ensure that the chunks stored in Qdrant are up-to-date.
    If not, update them accordingly.

    Args:
        chunks (list): List of chunks to store.
        embedding_model: The embedding model to use.
        collection_name (str): The name of the Qdrant collection.
    """
    print(colored("🔹 Syncing Chunks with Qdrant...", "yellow"))

    try:
        collections = qdrant_client.get_collections().collections
        collection_names = [col.name for col in collections]

        collection_exists = collection_name in collection_names

        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=collection_name,
            embedding=embedding_model
        )

        # Generate a new hash for the current chunks
        current_chunks_hash = compute_chunks_hash(chunks)

        # Try to fetch the previous hash stored in a dummy metadata
        previous_docs = query_qdrant("__hash_check__", embedding_model, collection_name, top_k=1) if collection_exists else []
        
        previous_hash = None
        if previous_docs:
            previous_hash = previous_docs[0].metadata.get("chunks_hash")

        # Check if the chunks are already up-to-date
        if previous_hash == current_chunks_hash:
            print(colored(f"✅ Chunks are already up-to-date in collection '{collection_name}'.", "green"))
            return
        else:
            if collection_exists:
                # Delete existing collection to replace it fully
                qdrant_client.delete_collection(collection_name=collection_name)
                print(colored(f"🔹 Deleted old collection '{collection_name}' to update with new chunks.", "cyan"))

            # Recreate collection
            dummy_embedding = embedding_model.embed_query("test")
            vector_size = len(dummy_embedding)

            qdrant_client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(colored(f"🔹 Recreated collection '{collection_name}' with updated chunks.", "cyan"))

            # Add documents (chunks)
            documents_to_add = []
            for chunk in chunks:
                doc = Document(
                    page_content=chunk['content'],
                    metadata={"source": chunk['source']}
                )
                documents_to_add.append(doc)

            # Add a dummy document to store the current chunks hash
            documents_to_add.append(Document(
                page_content="__hash_check__",
                metadata={"chunks_hash": current_chunks_hash}
            ))

            vector_store.add_documents(documents_to_add)
            print(colored(f"✅ Stored {len(documents_to_add)-1} chunks (and hash) in Qdrant collection '{collection_name}'.", "green"))

    except Exception as e:
        print(colored(f"❌ Error in syncing chunks: {str(e)}", "red"))

def query_qdrant(query: str, embedding_model, collection_name: str, top_k: int = 5) -> List[Document]:
    """
    Retrieve documents from Qdrant based on a query.
    
    Args:
        query (str): The query to retrieve documents for.
        embedding_model: The embedding model to use.
        collection_name (str): The name of the Qdrant collection.
        top_k (int): Number of documents to retrieve.
        
    Returns:
        List[Document]: List of retrieved documents.
    """
    print(colored(f"🔹 Querying Qdrant for: '{query}'", "yellow"))

    
    try:
        # Check if collection exists
        collections = qdrant_client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if collection_name not in collection_names:
            print(colored(f"❌ Collection '{collection_name}' doesn't exist in Qdrant.", "red"))
            return []
            
        # Create vector store for retrieval
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=collection_name,
            embedding=embedding_model
        )
        
        # Create retriever with search parameters
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )
        
        # Retrieve documents
        docs = retriever.invoke(query)
        
        print(colored(f"🔹 Retrieved {len(docs)} documents for query: '{query}'", "green"))
        return docs
        
    except Exception as e:
        print(colored(f"❌ Error while querying Qdrant: {str(e)}", "red"))
        return []

def format_retrieved_documents(docs: List[Document]) -> str:
    """
    Format retrieved documents into a readable string.
    
    Args:
        docs (List[Document]): List of retrieved documents.
        
    Returns:
        str: Formatted string of documents.
    """
    if not docs:
        return "No relevant documents found."
        
    formatted_docs = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown source")
        formatted_docs.append(f"Document {i}:\nSource: {source}\nContent: {doc.page_content}\n")
        
    return "\n".join(formatted_docs)

def get_context_for_llm(query: str, embedding_model, collection_name: str, top_k: int = 5) -> str:
    """
    Get context for LLM based on a query.
    
    Args:
        query (str): The query to get context for.
        embedding_model: The embedding model to use.
        collection_name (str): The name of the Qdrant collection.
        top_k (int): Number of documents to retrieve.
        
    Returns:
        str: Context for LLM.
    """
    docs = query_qdrant(query, embedding_model, collection_name, top_k)
    
    if not docs:
        return "No context available."
    
    # Create context string with relevant information
    context = "Here is the context from the documents:\n\n"
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown source")
        context += f"Document {i} (Source: {source}):\n{doc.page_content}\n\n"
    
    return context

