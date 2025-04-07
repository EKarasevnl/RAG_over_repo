import json
from tqdm import tqdm
from RAGsystem import RAGSystem


def evaluate(dataset_path='escrcpy-commits-generated.json'):
    """
    Evaluate the RAG system on a given dataset.
    
    :param dataset_path: The path to the dataset for evaluation (default: 'escrcpy-commits-generated.json').
    """
    rag = RAGSystem(index_exists=True)

    with open(dataset_path) as f:
        test_data = json.load(f)

    correct = 0
    for item in tqdm(test_data, desc="Evaluating"):
        if 'question' not in item or 'files' not in item:
            print(f"Skipping invalid item: {item}")
            continue

        results = [x[0] for x in rag.query(item['question'], top_k=10)]
        if any(f in results for f in item['files']):
            correct += 1

    print(f"Recall@10: {correct / len(test_data):.2%}")