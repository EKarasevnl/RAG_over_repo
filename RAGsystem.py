import os
import pickle
import re
from git import Repo
from collections import defaultdict
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util


class RAGSystem:
    def __init__(self, repo_url="https://github.com/viarotel-org/escrcpy", index_exists=False):
        """
        Initialize the RAGSystem with a repository URL and an optional flag for using an existing index.
        
        :param repo_url: The URL of the GitHub repository to index (default: 'https://github.com/viarotel-org/escrcpy').
        :param index_exists: Boolean flag to use an existing index (default: False).
        """
        self.biencoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.tokenizer = lambda text: re.findall(r"\w+", text.lower())

        if not index_exists and repo_url:
            self.repo_path = self.clone_repo(repo_url)
            self.chunks = self.process_files()
            self.build_index()
        elif index_exists:
            self.load_index()

    def clone_repo(self, url):
        """
        Clone the GitHub repository to a local directory.
        
        :param url: GitHub URL of the repository.
        :return: The path to the cloned repository.
        """
        repo_dir = "escrcpy_repo"
        if os.path.exists(repo_dir):
            print("Using existing repository")
            return repo_dir
        Repo.clone_from(url, repo_dir)
        return repo_dir

    def process_files(self, chunk_size=1000, overlap=50):
        """
        Process all the files in the cloned repository into chunks of text.

        :param chunk_size: The size of each chunk (default: 1000).
        :param overlap: The overlap between consecutive chunks (default: 50).
        :return: A list of chunks, each containing text, file path, line range, and file type.
        """
        chunks = []
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        try:
                            content = f.read()
                        except UnicodeDecodeError:
                            continue

                    lines = content.split('\n')
                    for i in range(0, len(lines), chunk_size - overlap):
                        chunk = '\n'.join(lines[i:i + chunk_size])
                        chunks.append({
                            'text': chunk,
                            'path': os.path.relpath(path, self.repo_path),
                            'lines': (i + 1, min(i + chunk_size, len(lines))),
                            'file_type': file.split('.')[-1] if '.' in file else ''
                        })
                except Exception as e:
                    print(f"Skipping {path} due to error: {str(e)}")
                    continue
        return chunks

    def build_index(self):
        """
        Build the BM25 index and sentence embeddings for all chunks.
        """
        self.corpus = [c['text'] for c in self.chunks]
        self.tokenized_corpus = [self.tokenizer(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.chunk_embeddings = self.biencoder.encode(self.corpus, show_progress_bar=True, convert_to_tensor=True)

        # Save the index and embeddings for later use
        with open('bm25.pkl', 'wb') as f:
            pickle.dump((self.chunks, self.tokenized_corpus, self.chunk_embeddings), f)

    def load_index(self):
        """
        Load an existing BM25 index and sentence embeddings from a saved file.
        """
        with open('bm25.pkl', 'rb') as f:
            self.chunks, self.tokenized_corpus, self.chunk_embeddings = pickle.load(f)
        self.corpus = [c['text'] for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def query(self, question, top_k=10):
        """
        Query the system with a question and return the top K most relevant files.

        :param question: The query question.
        :param top_k: The number of top results to return (default: 10).
        :return: A list of tuples containing the file path and its relevance score.
        """
        tokenized_query = self.tokenizer(question)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])[:1000]
        candidate_embeddings = self.chunk_embeddings[top_bm25_indices]

        query_embedding = self.biencoder.encode(question, convert_to_tensor=True)
        sim_scores = util.cos_sim(query_embedding, candidate_embeddings)[0]

        reranked = sorted(zip([self.chunks[i] for i in top_bm25_indices], sim_scores.tolist()), key=lambda x: -x[1])[:100]

        # Aggregate scores by file path
        file_scores = defaultdict(float)
        for chunk, score in reranked:
            file_scores[chunk['path']] += score

        return sorted(file_scores.items(), key=lambda x: -x[1])[:top_k]