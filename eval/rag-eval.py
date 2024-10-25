import pandas as pd
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
# from transformers import AutoTokenizer, AutoModelForSeq2SeqGeneration
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
from dataclasses import dataclass
import logging
from pathlib import Path

@dataclass
class SearchResult:
    chunk_id: str
    text: str
    score: float
    metadata: Dict = None

class RAGSystem:
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        generator_model_name: str = "google/flan-t5-large",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        # Loading the embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        if device == "cuda":
            self.embedding_model = self.embedding_model.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
        # self.generator = AutoModelForSeq2SeqGeneration.from_pretrained(generator_model_name).to(device)
        self.generator = T5ForConditionalGeneration.from_pretrained(generator_model_name).to(device)

        self.chunks = {}  # Dictionary to store chunks with their IDs
        self.chunk_embeddings = None
        self.chunk_ids = []

        logging.info(f"Initialized RAG system with device: {device}")

    def add_documents(self, documents: Dict[str, str]):
        """
        Add documents to the RAG system.
        documents: Dict with chunk_id as key and text as value
        """
        self.chunks = documents
        self.chunk_ids = list(documents.keys())

        # Compute embeddings for all chunks
        texts = [documents[chunk_id] for chunk_id in self.chunk_ids]
        self.chunk_embeddings = self.embedding_model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=True
        ).to(self.device)
        logging.info(f"Added {len(documents)} documents to the system")

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Retrieve the most relevant document chunks for a query
        """
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_tensor=True,
            show_progress_bar=False
        ).to(self.device)

        # Compute cosine similarity and ensure all embeddings are on the CPU for calculation
        similarity_scores = cosine_similarity(
            query_embedding.cpu().numpy().reshape(1, -1),
            self.chunk_embeddings.cpu().numpy()
        )[0]

        # Sort by similarity and get top_k results
        top_k_indices = np.argsort(similarity_scores)[-top_k:][::-1]

        results = []
        for idx in top_k_indices:
            chunk_id = self.chunk_ids[idx]
            results.append(SearchResult(
                chunk_id=chunk_id,
                text=self.chunks[chunk_id],
                score=float(similarity_scores[idx])
            ))

        return results

    def generate_answer(self, question: str, context: str) -> str:
        """
        Generate an answer using the retrieved context
        """
        prompt = f"Question: {question}\nContext: {context}\nAnswer:"

        inputs = self.tokenizer(
            prompt,
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.generator.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

    def query(self, question: str, top_k: int = 3) -> Dict:
        """
        End-to-end RAG pipeline: retrieve + generate
        """
        search_results = self.search(question, top_k=top_k)
        context = " ".join([result.text for result in search_results])
        answer = self.generate_answer(question, context)

        return {
            'answer': answer,
            'retrieved_chunks': [
                {'id': r.chunk_id, 'text': r.text, 'score': r.score}
                for r in search_results
            ]
        }

class GroundTruthEvaluator:
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system

    @staticmethod
    def load_ground_truth(file_path: str) -> List[Dict]:
        """
        Load ground truth data from JSON file
        """
        with open(file_path, 'r') as f:
            return json.load(f)

    def evaluate_retrieval(self, test_cases: List[Dict], top_k: int = 5) -> Dict:
        """
        Evaluate retrieval performance
        """
        precision_scores = []
        recall_scores = []
        mrr_scores = []

        for test_case in test_cases:
            question = test_case['question']
            relevant_chunks = set(test_case['relevant_chunks'])

            # Get retrievals from system
            results = self.rag_system.search(question, top_k=top_k)
            retrieved_chunks = [r.chunk_id for r in results]

            # Calculate metrics
            retrieved_relevant = set(retrieved_chunks) & relevant_chunks

            # Precision and Recall
            precision = len(retrieved_relevant) / len(retrieved_chunks) if retrieved_chunks else 0
            recall = len(retrieved_relevant) / len(relevant_chunks) if relevant_chunks else 0

            precision_scores.append(precision)
            recall_scores.append(recall)

            # MRR
            mrr = 0
            for i, chunk_id in enumerate(retrieved_chunks):
                if chunk_id in relevant_chunks:
                    mrr = 1.0 / (i + 1)
                    break
            mrr_scores.append(mrr)

        return {
            'precision@k': np.mean(precision_scores),
            'recall@k': np.mean(recall_scores),
            'mrr': np.mean(mrr_scores)
        }

    def evaluate_answer_quality(self, test_cases: List[Dict]) -> Dict:
        """
        Evaluate answer generation quality
        """
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        rouge_scores = []

        for test_case in test_cases:
            question = test_case['question']
            ground_truth = test_case['ground_truth_answer']

            # Generate answer
            response = self.rag_system.query(question)
            generated_answer = response['answer']

            # Calculate ROUGE scores
            scores = scorer.score(ground_truth, generated_answer)
            rouge_scores.append({
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            })

        # Average scores
        avg_scores = {
            'rouge1': np.mean([s['rouge1'] for s in rouge_scores]),
            'rouge2': np.mean([s['rouge2'] for s in rouge_scores]),
            'rougeL': np.mean([s['rougeL'] for s in rouge_scores])
        }

        return avg_scores

def main():
    # Initialize RAG system
    rag_system = RAGSystem()
    evaluator = GroundTruthEvaluator(rag_system)

    # Load ground truth data
    ground_truth = evaluator.load_ground_truth('ground_truth_dataset.json')

    # Load your document chunks
    document_chunks = {
        f"text_{i}": f"Content for chunk {i}"  # Replace with actual content
        for i in range(50)  # Adjust range based on your actual data
    }

    # Add documents to RAG system
    rag_system.add_documents(document_chunks)

    # Evaluate
    retrieval_metrics = evaluator.evaluate_retrieval(ground_truth)
    answer_metrics = evaluator.evaluate_answer_quality(ground_truth)

    # Save results
    results = {
        'retrieval_metrics': retrieval_metrics,
        'answer_metrics': answer_metrics
    }

    output_path = 'evaluation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print results
    print("\nRetrieval Metrics:")
    print("-----------------")
    for metric, value in retrieval_metrics.items():
        print(f"{metric}: {value:.3f}")

    print("\nAnswer Quality Metrics:")
    print("---------------------")
    for metric, value in answer_metrics.items():
        print(f"{metric}: {value:.3f}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
