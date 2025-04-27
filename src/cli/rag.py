from termcolor import colored
from src.app.LLM.gemini import get_answer_from_llm
from src.app.vector_store.vector_store import create_embeddings_in_qdrant, get_context_for_llm
from src.app.embedder.embedding import get_embedding
from src.app.chunker.website_chunking import create_chunks
from src.utils.functions import verifyUrl

#ENTRYPOINT
def rag():
    url = input(colored("Enter the website URL on which you want to query: ", "green"))

    domainName = verifyUrl(url)

    # Chunking
    chunks = create_chunks(url)
    
    # Embedding
    embedding = get_embedding()

    # Storing the chunks
    # create_embeddings_in_qdrant(chunks, embedding, collection_name=domainName)
    query = input(colored("Enter your query: ", "green"))

    retrieve = get_context_for_llm(query, embedding, collection_name=domainName)
    print(colored("🔹 Retrieved context: ", "yellow"))

    # Ask LLM
    response = get_answer_from_llm(query, retrieve)
    print(colored(f"🔹 LLM response: {response}", "yellow"))
    


