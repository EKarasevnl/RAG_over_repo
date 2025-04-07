import argparse
from RAGsystem import RAGSystem
from utils import evaluate


if __name__ == "__main__":
    """
    Entry point for the command-line interface.
    It handles actions like setting up the index, querying, and evaluating the system.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--setup', help="GitHub URL to index repository")
    parser.add_argument('--question', help="Ask a question")
    parser.add_argument('--evaluate', help="Specify the dataset for evaluation")
    args = parser.parse_args()

    if args.setup:
        # Build the index from the specified repository
        RAGSystem(repo_url=args.setup)
        print("Index built successfully")

    elif args.question:
        # Query the system with a question
        rag = RAGSystem(index_exists=True)
        print("Top relevant files:")
        for file, score in rag.query(args.question):
            print(f"- {file} (score: {score:.4f})")

    elif args.evaluate:
        # Evaluate the system on the provided dataset
        evaluate(args.evaluate)

    else:
        print("Please specify an action: --setup, --question, or --evaluate")
