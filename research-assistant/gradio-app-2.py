import gradio as gr
import os
from ingest import PDFProcessor, EmbeddingGenerator, store_in_lancedb
import pandas as pd
from rag import RAGSystem

# Function to create directories if they don't exist
def create_directories():
    if not os.path.exists('./uploaded_pdfs'):
        os.makedirs('./uploaded_pdfs')
    if not os.path.exists('./image_output'):
        os.makedirs('./image_output')

# Function to handle PDF file upload and generate embeddings
def process_pdf(pdf_file):
    if not pdf_file:
        return "No PDF file uploaded."

    try:
        # Use the pdf_file.name as it already points to the path
        pdf_path = pdf_file.name
        print(f"PDF path: {pdf_path}")

        # Process the PDF (text chunks + image extraction) using PDFProcessor
        pdf_processor = PDFProcessor(pdf_path)
        text_chunks = pdf_processor.extract_text_chunks()
        print(f"Text chunks extracted: {len(text_chunks)}")

        output_dir = "./image_output"  # Directory to save images
        pdf_processor.extract_images(output_dir)
        print(f"Images extracted to {output_dir}")

        # Generate embeddings using EmbeddingGenerator
        embedding_generator = EmbeddingGenerator()
        df = embedding_generator.create_embeddings_dataframe(text_chunks, output_dir)
        print(f"Embeddings dataframe created with {len(df)} rows")

        # Store embeddings in LanceDB
        table = store_in_lancedb(df)
        print(f"Embeddings stored in LanceDB")

    except Exception as e:
        print(f"Error during PDF processing: {e}")
        return f"Error: {str(e)}"

    return f"Embeddings generated and stored in LanceDB. {len(df)} rows processed."

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

# Function to handle question querying
def query_question(question, top_k, model):
    answer, sources = rag_pipeline(question, top_k, model)
    return answer, sources

# Gradio interface for the RAG application with PDF upload
def create_gradio_app():
    # Define the Gradio interface
    with gr.Blocks() as interface:
        # PDF file upload and embedding generation
        gr.Markdown("# RAG-based Research Assistant")
        gr.Markdown("## Ask questions and get answers based on relevant research content.")
        gr.Markdown("### Steps:")
        gr.Markdown("#### 1. Drop or upload your research paper in PDF format.")
        gr.Markdown("#### 2. Click `Generate Embeddings` button. This may take a while...")
        gr.Markdown("#### 3. Ask your question. Optionally, select number of top results and LLM model.")
        gr.Markdown("#### 3. Click `Get Answer` button")


        pdf_file = gr.File(label="Upload PDF file")
        upload_button = gr.Button("Generate Embeddings")
        embedding_status = gr.Textbox(label="Embedding Status:", interactive=False)

        def upload_pdf_action(pdf_file):
            return process_pdf(pdf_file)

        upload_button.click(upload_pdf_action, inputs=[pdf_file], outputs=[embedding_status])

        # Question answering section
        question = gr.Textbox(lines=2, label="Ask a question")
        top_k = gr.Slider(1, 5, value=3, step=1, label="Number of top results")
        model = gr.Dropdown(choices=['meta-llama/Llama-Vision-Free'], label="LLM Model")

        query_button = gr.Button("Get Answer")

        answer_output = gr.Textbox(label="Generated Answer")
        sources_output = gr.Textbox(label="Relevant Sources")

        query_button.click(query_question, inputs=[question, top_k, model], outputs=[answer_output, sources_output])

    return interface


# Launch the app
if __name__ == "__main__":
    create_directories()  # Ensure directories exist before running the app
    app = create_gradio_app()
    app.launch()
