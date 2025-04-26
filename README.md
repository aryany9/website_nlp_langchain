```
rag_scraper/
│
├── app/                         # Main application logic
│   ├── __init__.py
│   ├── scraper/                  # Scraping logic
│   │   ├── __init__.py
│   │   ├── scraper.py            # Scrape website, get HTML, routing URLs
│   │   └── utils.py              # HTML cleaning, URL normalization etc.
│   │
│   ├── chunker/                  # Chunking logic
│   │   ├── __init__.py
│   │   └── chunker.py            # HTML/Text chunking functions
│   │
│   ├── embedder/                 # Embedding generation
│   │   ├── __init__.py
│   │   └── embedder.py           # Gemini Langchain embedder wrapper
│   │
│   ├── vector_store/             # Vector database interactions
│   │   ├── __init__.py
│   │   └── qdrant_client.py      # Functions to upsert, query Qdrant
│   │
│   ├── semantic_router/          # Logical routing based on queries
│   │   ├── __init__.py
│   │   └── router.py             # System prompt logic / routing rules
│   │
│   └── query_engine/             # Answer user queries
│       ├── __init__.py
│       └── query_processor.py    # Querying vector DB, synthesizing answers
│
├── config/                       # Configurations
│   ├── __init__.py
│   └── settings.py               # Environment variables, API keys, etc.
│
├── cli/                          # Terminal CLI interface
│   ├── __init__.py
│   └── main.py                   # CLI app entry point
│
├── tests/                        # Unit and integration tests
│   ├── __init__.py
│   ├── test_scraper.py
│   ├── test_chunker.py
│   ├── test_embedder.py
│   └── etc...
│
├── .env                           # Environment variables (for local dev)
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── pyproject.toml                 # (Optional) For better packaging
```