from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client import QdrantClient
from termcolor import colored
from typing import List, Optional, Dict, Any

def create_embeddings_in_qdrant(chunks: list, embedding_model, collection_name: str):
    """
    Store chunks in Qdrant with embeddings.

    Args:
        chunks (list): List of chunks to store.
        embedding_model: The embedding model to use.
        collection_name (str): The name of the Qdrant collection.
    """
    print(colored("🔹 Storing Chunks in Qdrant  ", "yellow"))

    # Connect to existing Qdrant instance
    qdrant_client = QdrantClient(
        url="http://localhost:6333",
    )
    
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
    
    # Connect to Qdrant
    qdrant_client = QdrantClient(
        url="http://localhost:6333",
    )
    
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