import os
import json
import bs4
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup
from termcolor import colored

from cli.spinner import spinner

@spinner("Creating Chunks")
def create_chunks(url: str, max_chunk_size: int = 1000, overlap_size: int = 200) -> list:
    """
    Chunk a website into smaller parts for processing.

    Args:
        url (str): The URL of the website to chunk.
        max_chunk_size (int): The maximum size of each chunk.
        overlap_size (int): The overlap between chunks.

    Returns:
        list: A list of chunks from the website.
    """
    from urllib.parse import urlparse

    # Create 'chunks' directory if it doesn't exist
    chunks_folder = os.path.join(os.path.dirname(__file__), "chunks")
    os.makedirs(chunks_folder, exist_ok=True)

    # Generate safe filename from domain
    parsed_url = urlparse(url)
    domain_name = parsed_url.netloc.replace('.', '_')
    chunk_file_path = os.path.join(chunks_folder, f"{domain_name}.json")

    # If file already exists, load and return chunks
    if os.path.exists(chunk_file_path):
        print(colored(f"🔹 Loading existing chunks from {chunk_file_path}", "cyan"))
        with open(chunk_file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(colored(f"🔹 Loaded {len(chunks)} chunks from cache.", "green"))
        return chunks

    # If not found, create fresh chunks
    print(colored("🔹 Creating Chunks from Website", "yellow"))

    # Load website content
    loader = RecursiveUrlLoader(
        url=url,
        max_depth=1000,
        extractor=lambda x: Soup(x, "html.parser").text
    )
    documents = loader.load()

    # Print all the URLs from which documents are loaded
    print(colored("🔹 URLs Found:", "cyan"))
    for idx, doc in enumerate(documents, start=1):
        url_from_doc = doc.metadata.get('source', 'Unknown URL')
        print(f"{idx}. {url_from_doc}")

    print(colored(f"\n🔹 Loaded {len(documents)} documents from the website.", "green"))

    # Create chunks per page
    all_chunks = []
    for idx, doc in enumerate(documents, start=1):
        url_from_doc = doc.metadata.get('source', 'Unknown URL')
        page_text = doc.page_content

        soup = Soup(page_text, "html.parser")
        cleaned_text = soup.get_text()

        # Split into chunks with overlap
        start = 0
        while start < len(cleaned_text):
            end = start + max_chunk_size
            chunk = cleaned_text[start:end]
            all_chunks.append({
                "content": chunk.strip(),
                "source": url_from_doc
            })
            start = end - overlap_size  # Move back by overlap size for overlap

        print(colored(f"🔹 Page {idx} ({url_from_doc}): {len(all_chunks)} chunks created so far.", "magenta"))

    print(colored(f"\n🔹 Total {len(all_chunks)} chunks created from the website.", "green"))

    # Save the chunks to file for future
    with open(chunk_file_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(colored(f"🔹 Chunks saved to {chunk_file_path}", "cyan"))

    return all_chunks
