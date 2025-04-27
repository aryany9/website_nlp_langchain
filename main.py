from dotenv import load_dotenv
from src.cli.rag import rag
import sys


def main():
    load_dotenv()
    sys.path.append("src")
    rag()

if __name__ == "__main__":
    main()