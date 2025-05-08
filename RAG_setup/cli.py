# cli.py
import argparse
from chat_pdf import QA

def main():
    parser = argparse.ArgumentParser(description="RAG ChatPDF CLI using DeepSeek + Ollama")
    parser.add_argument("--pdf", type=str, help="Path to PDF file to ingest", required=True)
    parser.add_argument("--question", type=str, help="Question to ask", required=True)
    parser.add_argument("--k", type=int, default=5, help="Number of retrieved docs (default: 5)")
    parser.add_argument("--threshold", type=float, default=0.2, help="Score threshold (default: 0.2)")

    args = parser.parse_args()

    assistant = QA()
    print(f" Ingesting document: {args.pdf}")
    assistant.ingest(args.pdf)

    print(f"\n Question: {args.question}")
    answer = assistant.ask(args.question, k=args.k, score_threshold=args.threshold)

    print(f"\n Answer:\n{answer}")


if __name__ == "__main__":
    main()
