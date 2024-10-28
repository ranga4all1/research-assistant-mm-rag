import gradio as gr
import os
import shutil
from ingest import PDFProcessor, EmbeddingGenerator, store_in_lancedb
import pandas as pd
from rag import RAGSystem
from monitoring import setup_rag_monitoring

# Function to create directories if they don't exist
def create_directories():
    if not os.path.exists('./uploaded_pdfs'):
        os.makedirs('./uploaded_pdfs')
    if not os.path.exists('./image_output'):
        os.makedirs('./image_output')

def process_pdf(pdf_file):
    if not pdf_file:
        return "No PDF file uploaded."

    try:
        # Save the uploaded PDF to the correct directory
        pdf_filename = os.path.basename(pdf_file.name)
        pdf_output_path = f"./uploaded_pdfs/{pdf_filename}"
        shutil.copyfile(pdf_file.name, pdf_output_path)
        print(f"PDF saved to: {pdf_output_path}")

        # Process the PDF (text chunks + image extraction)
        pdf_processor = PDFProcessor(pdf_output_path)
        text_chunks = pdf_processor.extract_text_chunks()
        print(f"Text chunks extracted: {len(text_chunks)}")

        # Ensure images are saved to the correct folder
        output_dir = "./image_output"
        pdf_processor.extract_images(output_dir)
        print(f"Images extracted to {output_dir}")

        # Generate embeddings
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

# Initialize the RAG system with monitoring
rag_system = RAGSystem()
monitor = setup_rag_monitoring(rag_system)

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


def show_performance_report():
    report_path = './monitoring/performance_report.html'
    if os.path.exists(report_path):
        # Temporarily using a small HTML snippet for testing
        test_content = "<h1>Detailed report:</h1><p>Available in the monitoring folder (./monitoring/performance_report) on your system.</p>"
        return test_content
    else:
        return "Report file not found."


def create_gradio_app():
    # Initialize the RAG system with monitoring
    rag_system = RAGSystem()
    monitor = setup_rag_monitoring(rag_system)

    with gr.Blocks() as interface:
        # Main title
        gr.Markdown("# RAG-based Research Assistant")

        with gr.Tab("Document Upload"):
            pdf_file = gr.File(label="Upload PDF file")
            upload_button = gr.Button("Generate Embeddings")
            embedding_status = gr.Textbox(label="Embedding Status:", interactive=False)

            def upload_pdf_action(pdf_file):
                return process_pdf(pdf_file)

            upload_button.click(upload_pdf_action, inputs=[pdf_file], outputs=[embedding_status])

        with gr.Tab("Query System"):
            # Question answering section
            question = gr.Textbox(lines=2, label="Ask a question:")
            with gr.Row():
                top_k = gr.Slider(1, 5, value=3, step=1, label="Number of top results:")
                model = gr.Dropdown(choices=['meta-llama/Llama-Vision-Free'], label="LLM Model:")

            query_button = gr.Button("Get Answer")
            answer_output = gr.Textbox(label="Generated Answer:")
            sources_output = gr.Textbox(label="Relevant Sources:")

        with gr.Tab("System Monitoring"):
            refresh_button = gr.Button("Refresh Monitoring Stats")

            with gr.Row():
                with gr.Column():
                    query_stats = gr.JSON(label="Query Statistics")
                with gr.Column():
                    retrieval_stats = gr.JSON(label="Retrieval Quality")
                with gr.Column():
                    diversity_stats = gr.JSON(label="Retrieval Diversity")

            def refresh_monitoring():
                stats = monitor.get_metrics_summary()
                query_data = {
                    'Total Queries': stats['total_queries'],
                    'Queries (Last 24h)': stats['queries_last_24h'],
                    'Avg Answer Length': stats['avg_answer_length']
                }
                retrieval_data = {
                    'Avg Score': stats['avg_retrieval_score'],
                    'Min Score': stats['retrieval_quality']['min_score'],
                    'Max Score': stats['retrieval_quality']['max_score'],
                    'Score Std Dev': stats['retrieval_quality']['score_std']
                }
                diversity_data = {
                    'Avg Documents': stats['avg_num_docs'],
                    'Avg Diversity': stats['diversity']['avg_diversity']
                }
                return query_data, retrieval_data, diversity_data

            refresh_button.click(
                refresh_monitoring,
                outputs=[query_stats, retrieval_stats, diversity_stats]
            )

            # Link to detailed report
            # gr.Markdown("""
            #     ### Detailed Monitoring Reports
            #     - View the full performance report in the monitoring folder.)
            # """)

            # Inline report display
            report_button = gr.Button("View Detailed Report")
            report_output = gr.HTML(label="Performance Report")

            report_button.click(
                show_performance_report,
                outputs=report_output
            )

        def query_with_monitoring(question, top_k, model):
            response = rag_system.query(question, top_k, model)
            return response['answer'], '\n'.join([
                f"- {result['type'].capitalize()} content (Page {result['page_num']}, Score: {result['score']:.2f})"
                for _, result in response['search_results'].iterrows()
            ])

        query_button.click(
            query_with_monitoring,
            inputs=[question, top_k, model],
            outputs=[answer_output, sources_output]
        )

    return interface

if __name__ == "__main__":
    if not os.path.exists('./uploaded_pdfs'):
        os.makedirs('./uploaded_pdfs')
    if not os.path.exists('./image_output'):
        os.makedirs('./image_output')

    app = create_gradio_app()
    app.launch()
