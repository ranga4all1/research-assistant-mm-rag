# RAG Flow and Retrieval evaluation

## Evaluation Metrics Implemented:

- For Retrieval:
    - Precision@k
    - Recall@k
    - MRR (Mean Reciprocal Rank)

- For Answer Quality:
    - ROUGE scores
        - ROUGE-1
        - ROUGE-2
        - ROUGE-L


## Evaluation results

- Retrieval Metrics:
   - precision@k: 0.050
   - recall@k: 0.083
   - mrr: 0.156

- Answer Quality Metrics:
   - rouge1: 0.024
   - rouge2: 0.006
   - rougeL: 0.024


## Eval results explanation:

Theese metrics reflect the effectiveness of our retrieval-augmented generation (RAG) system in terms of retrieval relevance and answer quality. Here’s what each of these scores indicates in our context:

### Retrieval Metrics
These measure how well the system retrieves relevant document chunks for a given query.

- Precision@k (0.050): This is the proportion of retrieved chunks that are relevant among the top k results (e.g., top 5 results if k=5). A score of 0.050 (or 5%) indicates that only a small portion of the retrieved chunks are relevant to the query, suggesting that the retrieval model may need tuning to improve accuracy.

- Recall@k (0.083): This measures how many of the relevant chunks are retrieved within the top k results. A score of 0.083 (or 8.3%) means the system is finding only a small fraction of all relevant chunks, implying that it misses many relevant documents.

- MRR (Mean Reciprocal Rank) (0.156): MRR considers the position of the first relevant chunk in the retrieved results. A score of 0.156 (or ~15.6%) means that relevant chunks often appear further down the ranked list, showing that the most relevant information isn’t frequently in the top-ranked positions.

These low retrieval scores indicate the model is struggling to identify and rank the most relevant chunks accurately.

## Answer Quality Metrics
These measure the quality of generated answers using ROUGE, a common metric for comparing text similarity based on overlapping n-grams.

- ROUGE-1 (0.024): This score represents the overlap of unigrams (single words) between the generated answers and the ground truth. A low score of 0.024 suggests that there is very little lexical similarity between the generated answers and the expected answers.

- ROUGE-2 (0.006): This evaluates bigram (two-word sequence) overlap, capturing more specific similarity. The score 0.006 reflects even less overlap in meaningful word sequences, meaning the generated answers are not closely matching the structure or content of the ground truth.

- ROUGE-L (0.024): This measures the longest common subsequence, assessing how well the model captures the overall structure of the reference answer. A score of 0.024 indicates the generated answers do not align well with the reference answers in terms of structure.

These low ROUGE scores indicate that the generated answers are not closely aligned with the expected answers, which could be due to poor retrieval results, ineffective use of context, or suboptimal answer generation by the language model.

## Recommendations for Improvement in next iterations of the project
To improve these metrics:

1. Enhance Retrieval Accuracy:

Fine-tune the embedding model on domain-specific data.
Experiment with other retrieval models or embedding configurations to better capture semantic relevance.

2. Refine Context Aggregation:

Instead of using a simple concatenation of top results as context, consider strategies like ranking based on semantic similarity to increase answer relevance.

3. Improve Answer Generation:

Try different hyperparameters for the language model, such as top_k, temperature, or prompt engineering to focus more on precision.
Consider fine-tuning the language model if feasible for better domain-specific answer generation.