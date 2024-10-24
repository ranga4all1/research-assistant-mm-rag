import gradio as gr
import pandas as pd
from rag import RAGSystem  # Assuming the rag.py is named as rag.py and is in the same directory.

# Initialize the RAG system
rag_system = RAGSystem()

def rag_pipeline(question, top_k=5, model='meta-llama/Llama-Vision-Free'):
    # Use the RAG system to query and get results
    response = rag_system.query(question, top_k=top_k, model=model)

    # Extract the answer and the relevant sources
    answer = response['answer']
    search_results = response['search_results']

    # Format the search results into a readable form
    results_str = ""
    for _, result in search_results.iterrows():
        results_str += f"- {result['type'].capitalize()} content (Page {result['page_num']}, Score: {result['score']:.2f})\n"

    return answer, results_str

# Define Gradio interface
def gradio_interface(question, top_k, model):
    answer, sources = rag_pipeline(question, top_k, model)
    return answer, sources

# Gradio UI setup
interface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(lines=2, label="Ask a question"),
        gr.Slider(1, 5, step=1, value=3, label="Number of top results"),
        gr.Dropdown(choices=['meta-llama/Llama-Vision-Free'], label="LLM Model")
    ],
    outputs=[
        gr.Textbox(label="Generated Answer"),
        gr.Textbox(label="Relevant Sources")
    ],
    title="RAG-based Research Assistant",
    description="Ask questions and get answers based on relevant research content.",
)

# Launch the app
interface.launch()
