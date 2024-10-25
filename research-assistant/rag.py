import os
import numpy as np
import lancedb
from dotenv import load_dotenv
from together import Together
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor


class EmbeddingResizer:
    """Utility class to resize embeddings to 768 dimensions"""
    @staticmethod
    def resize_to_768(embedding):
        embedding = np.array(embedding)
        current_dim = embedding.shape[0]

        if current_dim == 768:
            return embedding

        if current_dim > 768:
            return embedding[:768]
        else:
            padding = np.zeros(768 - current_dim)
            return np.concatenate([embedding, padding])

class RAGSystem:
    def __init__(self, db_path="./lancedb"):
        # Load environment variables and set API key
        load_dotenv()
        Together.api_key = os.getenv("TOGETHER_API_KEY")

        # Initialize Together AI client
        self.client = Together()

        # Initialize LanceDB
        self.db = lancedb.connect(db_path)
        self.table = self.db.open_table("multimodal_embeddings")

        # Initialize embedding models
        self.text_model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
        self.resizer = EmbeddingResizer

    def get_query_embedding(self, query):
        """Generate embedding for the query text"""
        query_embedding = self.text_model.encode(query,
                                               convert_to_numpy=True,
                                               normalize_embeddings=True)
        query_embedding = self.resizer.resize_to_768(query_embedding)
        return query_embedding / np.linalg.norm(query_embedding)

    def search(self, query, top_k=5):
        """Search for relevant documents using the query"""
        query_embedding = self.get_query_embedding(query)
        results = self.table.search(query_embedding).limit(top_k).to_pandas()
        results['score'] = 1 - results['_distance']
        return results

    def build_prompt(self, query, search_results):
        """Build the prompt for the LLM"""
        context_template = """
            Content: {content}
            Type: {type}
            Page: {page_num}
            Relevance Score: {score:.2f}
        """.strip()

        context = "\n\n".join([
            context_template.format(**row.to_dict())
            for _, row in search_results.iterrows()
        ])

        prompt = f"""You're a research assistant helping users understand academic papers.
            Answer the following question based only on the provided context from our research papers database.
            Use only the facts from the context when answering.

            Question: {query}

            Context:
            {context}

            Answer:"""

        return prompt

    def generate_answer(self, prompt, model='meta-llama/Llama-Vision-Free'):
        """Generate answer using the LLM"""
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def query(self, question, top_k=5, model='meta-llama/Llama-Vision-Free'):
        """Complete RAG pipeline"""
        # Search for relevant content
        search_results = self.search(question, top_k=top_k)

        # Build prompt with context
        prompt = self.build_prompt(question, search_results)

        # Generate answer
        answer = self.generate_answer(prompt, model=model)

        return {
            'answer': answer,
            'search_results': search_results,
            'prompt': prompt
        }

def main():

    # Initialize RAG system
    rag_system = RAGSystem()

    # Example usage
    query = "What is a transformer?"
    response = rag_system.query(query)

    print("\nQuestion:", query)
    print("\nAnswer:", response['answer'])
    print("\nTop relevant sources:")
    for _, result in response['search_results'].iterrows():
        print(f"- {result['type'].capitalize()} content (Page {result['page_num']}, Score: {result['score']:.2f})")

if __name__ == "__main__":
    main()
