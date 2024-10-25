import numpy as np
import pandas as pd
import io
import os
import pymupdf
from PIL import Image
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor
import torch
import torch.nn.functional as F
import lancedb
import pyarrow as pa

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

class PDFProcessor:
    def __init__(self, pdf_path, chunk_size=1000, overlap_size=200):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.pdf_document = pymupdf.open(pdf_path)

    def extract_text_chunks(self):
        chunks = []
        for page_num in range(len(self.pdf_document)):
            page = self.pdf_document[page_num]
            page_text = page.get_text()

            start = 0
            while start < len(page_text):
                end = start + self.chunk_size
                chunk = page_text[start:end]
                chunks.append((page_num + 1, chunk))
                start += self.chunk_size - self.overlap_size
        return chunks

    def extract_images(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for page_index in range(len(self.pdf_document)):
            page = self.pdf_document[page_index]
            image_list = page.get_images()

            if image_list:
                print(f"[+] Found {len(image_list)} images in page {page_index}")

            for image_index, img in enumerate(page.get_images(), start=1):
                xref = img[0]
                base_image = self.pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image = Image.open(io.BytesIO(image_bytes))
                # image.save(open(f"{output_dir}image{page_index+1}_{image_index}.{image_ext}", "wb"))
                image.save(open(f"{output_dir}/image{page_index+1}_{image_index}.{image_ext}", "wb"))

        print(f"Image extraction completed. Saved images to {output_dir}")

class EmbeddingGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text_model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
        self.image_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.resizer = EmbeddingResizer

    def create_embeddings_dataframe(self, text_chunks, image_folder):
        combined_data = []

        # Process text chunks
        print("Processing text chunks...")
        text_embeddings = self.text_model.encode([chunk[1] for chunk in text_chunks],
                                               show_progress_bar=True,
                                               convert_to_numpy=True,
                                               normalize_embeddings=True)

        for idx, (chunk, embedding) in enumerate(zip(text_chunks, text_embeddings)):
            embedding = self.resizer.resize_to_768(embedding)
            embedding = embedding / np.linalg.norm(embedding)
            combined_data.append({
                'id': f'text_{idx}',
                'content': chunk[1],
                'page_num': chunk[0],
                'embedding': embedding,
                'type': 'text'
            })

        # Process images
        print("Processing images...")
        image_files = [f for f in os.listdir(image_folder)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if image_files:
            images = []
            valid_image_files = []

            for image_file in tqdm(image_files, desc="Loading images"):
                image_path = os.path.join(image_folder, image_file)
                try:
                    image = Image.open(image_path).convert('RGB')
                    images.append(image)
                    valid_image_files.append(image_path)
                except Exception as e:
                    print(f"Error loading image {image_file}: {str(e)}")
                    continue

            if images:
                batch_size = 32
                all_embeddings = []

                for i in range(0, len(images), batch_size):
                    batch_images = images[i:i + batch_size]
                    inputs = self.processor(images=batch_images, return_tensors="pt", padding=True).to(self.device)

                    with torch.no_grad():
                        outputs = self.image_model.get_image_features(**inputs)
                        batch_embeddings = outputs.detach().cpu().numpy()
                        all_embeddings.extend(batch_embeddings)

                start_idx = len(combined_data)
                for idx, (image_path, embedding) in enumerate(zip(valid_image_files, all_embeddings)):
                    embedding = self.resizer.resize_to_768(embedding)
                    embedding = embedding / np.linalg.norm(embedding)
                    combined_data.append({
                        'id': f'image_{start_idx + idx}',
                        'content': image_path,
                        'page_num': int(image_path.split('image')[-1].split('_')[0]),
                        'embedding': embedding,
                        'type': 'image'
                    })

        df = pd.DataFrame(combined_data)
        print(f"Created DataFrame with {len(df)} rows ({len(text_chunks)} text + {len(images) if 'images' in locals() else 0} images)")
        return df

def store_in_lancedb(df, db_path="./lancedb"):
    embedding_dimension = 768
    schema = pa.schema([
        ('id', pa.string()),
        ('content', pa.string()),
        ('page_num', pa.int32()),
        ('embedding', pa.list_(pa.float32(), embedding_dimension)),
        ('type', pa.string()),
    ])

    db = lancedb.connect(db_path)
    df['content'] = df['content'].astype(str)

    table = db.create_table(
        "multimodal_embeddings",
        data=df,
        schema=schema,
        mode="overwrite"
    )

    table.create_index(
        vector_column_name="embedding",
        index_type="IVF_HNSW_SQ",
        num_partitions=8,
        num_sub_vectors=2
    )

    return table

def main():
    # Configuration
    pdf_path = '../data/raw/attention.pdf'
    output_dir = "../data/processed/"

    # Process PDF
    pdf_processor = PDFProcessor(pdf_path)
    text_chunks = pdf_processor.extract_text_chunks()
    pdf_processor.extract_images(output_dir)

    # Generate embeddings
    embedding_generator = EmbeddingGenerator()
    df = embedding_generator.create_embeddings_dataframe(text_chunks, output_dir)

    # Store in LanceDB
    table = store_in_lancedb(df)
    print("Data ingestion completed successfully!")

if __name__ == "__main__":
    main()
