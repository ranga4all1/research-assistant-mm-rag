import pymupdf
import json
import numpy as np
from typing import List, Dict
from tqdm.auto import tqdm
import re


class PDFGroundTruthGenerator:
    def __init__(self, pdf_path: str, chunk_size: int = 1000, overlap_size: int = 200):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.pdf_document = None
        self.text_chunks = {}
        self.load_pdf()

    def load_pdf(self):
        """Load PDF using PyMuPDF"""
        self.pdf_document = pymupdf.open(self.pdf_path)

    def create_chunks(self):
        """Create overlapping chunks of text from PDF content"""
        chunk_id = 0

        for page_num in range(len(self.pdf_document)):
            page = self.pdf_document[page_num]
            page_text = page.get_text()

            # Clean the text
            page_text = re.sub(r'\s+', ' ', page_text).strip()

            # Create chunks with overlap
            start = 0
            while start < len(page_text):
                end = start + self.chunk_size
                chunk = page_text[start:end]

                if chunk.strip():
                    chunk_id_str = f"text_{chunk_id}"
                    self.text_chunks[chunk_id_str] = {
                        'text': chunk,
                        'page_number': page_num + 1,  # 1-based page numbering
                        'start_char': start,
                        'end_char': end
                    }
                    chunk_id += 1

                start += self.chunk_size - self.overlap_size

    def find_relevant_chunks(self, question: str, answer: str) -> List[Dict]:
        """Find relevant chunks using keyword matching and context"""
        relevant_chunks = []

        # Extract keywords from question and answer
        keywords = set(re.findall(r'\w+', question.lower()) +
                      re.findall(r'\w+', answer.lower()))

        # Remove common stop words
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with'}
        keywords = keywords - stop_words

        for chunk_id, chunk_data in self.text_chunks.items():
            chunk_text = chunk_data['text'].lower()

            # Count keyword matches
            matched_keywords = sum(1 for keyword in keywords
                                if keyword in chunk_text)

            # Calculate keyword density
            keyword_density = matched_keywords / len(chunk_text.split())

            # Check for answer phrase matches
            answer_phrases = [phrase.strip().lower()
                            for phrase in answer.split('.') if phrase.strip()]
            phrase_matches = sum(1 for phrase in answer_phrases
                               if phrase in chunk_text)

            # Determine if chunk is relevant based on multiple criteria
            if (matched_keywords >= 3 or  # Multiple keyword matches
                keyword_density > 0.1 or   # High keyword density
                phrase_matches > 0):       # Contains answer phrases
                relevant_chunks.append({
                    'chunk_id': chunk_id,
                    'page_number': chunk_data['page_number'],
                    'relevance_score': matched_keywords + (phrase_matches * 2)
                })

        # Sort by relevance score and take top N chunks
        relevant_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevant_chunks[:3]  # Limit to top 3 most relevant chunks

    def generate_ground_truth(self, qa_pairs: List[Dict]) -> List[Dict]:
        """Generate ground truth dataset with relevant chunks and page numbers"""
        if not self.text_chunks:
            self.create_chunks()

        ground_truth_data = []

        for qa_pair in tqdm(qa_pairs, desc="Generating ground truth"):
            question = qa_pair['question']
            answer = qa_pair['answer']

            # Find relevant chunks
            relevant_chunks = self.find_relevant_chunks(question, answer)

            ground_truth_entry = {
                'question': question,
                'ground_truth_answer': answer,
                'relevant_chunks': [chunk['chunk_id'] for chunk in relevant_chunks],
                'page_numbers': sorted(list(set(chunk['page_number']
                                             for chunk in relevant_chunks)))
            }

            ground_truth_data.append(ground_truth_entry)

        return ground_truth_data


def main():
    # Define QA pairs
    qa_pairs = [
        # Architecture Overview
        {
            'question': 'What is the key innovation of the transformer architecture?',
            'answer': 'The key innovation is the self-attention mechanism that directly models relationships between all words in a sequence, regardless of their position, allowing the model to process all positions simultaneously.'
        },
        {
            'question': 'What are the three main components of the Transformer architecture?',
            'answer': 'The three main components are the encoder stack, decoder stack, and attention mechanisms (self-attention and encoder-decoder attention).'
        },

        # Attention Mechanism
        {
            'question': 'How does self-attention work in the Transformer?',
            'answer': 'Self-attention computes attention weights by using queries, keys, and values derived from the input sequence. It calculates compatibility between each position and all other positions, allowing the model to weigh the importance of different parts of the input when processing each position.'
        },
        {
            'question': 'What is multi-head attention and why is it used?',
            'answer': 'Multi-head attention runs multiple attention mechanisms in parallel, with different learned linear transformations of queries, keys, and values. This allows the model to attend to information from different representation subspaces at different positions, capturing different types of relationships.'
        },

        # Model Details
        {
            'question': 'What is the dimension of the model in the paper\'s base configuration?',
            'answer': 'The base model uses d_model = 512, with 6 encoder and decoder layers, and 8 attention heads.'
        },
        {
            'question': 'How does the Transformer handle positional information?',
            'answer': 'The Transformer uses sinusoidal positional encodings added to the input embeddings to inject information about the relative or absolute position of tokens in the sequence.'
        },

        # Training and Performance
        {
            'question': 'What are the advantages of the Transformer over RNNs and CNNs?',
            'answer': 'The Transformer reduces sequential computation, allows for more parallelization, and has shorter path lengths between long-range dependencies. It achieves better performance while being more parallelizable and requiring significantly less time to train.'
        },
        {
            'question': 'What training optimizations are used in the Transformer?',
            'answer': 'The model uses label smoothing during training, the Adam optimizer with custom learning rate scheduling (warmup and decay), dropout, and residual connections.'
        },

        # Technical Details
        {
            'question': 'How is the attention score calculated in the Transformer?',
            'answer': 'The attention score is calculated as Attention(Q,K,V) = softmax(QK^T/√dk)V, where Q is queries, K is keys, V is values, and dk is the dimension of the keys.'
        },
        {
            'question': 'What is the purpose of the scaling factor in the attention mechanism?',
            'answer': 'The scaling factor (1/√dk) is used to counteract the effect of the dot products growing large in magnitude for large values of dk, which could push the softmax function into regions with extremely small gradients.'
        },

        # Architecture Components
        {
            'question': 'What is the role of the feed-forward networks in the Transformer?',
            'answer': 'The feed-forward networks in each encoder and decoder layer consist of two linear transformations with ReLU activation in between, processing each position independently and identically. They allow the model to process the attended information and introduce non-linearity.'
        },
        {
            'question': 'How does the decoder prevent attention to subsequent positions?',
            'answer': 'The decoder uses masked multi-head attention in its self-attention layer, which prevents positions from attending to subsequent positions by masking out (setting to -infinity) all values in the input of the softmax which correspond to illegal connections.'
        },

        # Results and Applications
        {
            'question': 'What tasks was the Transformer evaluated on?',
            'answer': 'The Transformer was evaluated on machine translation tasks (WMT 2014 English-to-German and English-to-French translation) and showed superior performance while being more parallelizable and requiring significantly less time to train.'
        },
        {
            'question': 'What is the computational complexity of self-attention compared to recurrent layers?',
            'answer': 'Self-attention has a complexity of O(n²·d) where n is sequence length and d is representation dimension, while recurrent layers have complexity O(n·d²). Self-attention is faster for sequences shorter than the representation dimensionality.'
        },

        # Implementation Details
        {
            'question': 'What is the purpose of residual connections in the Transformer?',
            'answer': 'Residual connections are used around each sub-layer (self-attention and feed-forward networks), followed by layer normalization. They help with training deeper networks by allowing gradients to flow directly through the network.'
        },
        {
            'question': 'How does the Transformer handle varying sequence lengths?',
            'answer': 'The Transformer handles varying sequence lengths through padding and masking. The attention mechanism can be implemented to mask out padded positions, ensuring they don\'t contribute to the attention calculations.'
        }
    ]

    # Initialize the generator
    generator = PDFGroundTruthGenerator(
        pdf_path='../data/raw/attention.pdf',
        chunk_size=1000,
        overlap_size=200
    )

    # Generate ground truth dataset
    ground_truth_data = generator.generate_ground_truth(qa_pairs)

    # Save to JSON file
    with open('ground_truth_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(ground_truth_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
