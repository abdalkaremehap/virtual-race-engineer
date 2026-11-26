import os

import PyPDF2
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer


def extract_text_from_pdf(pdf_path):
    logger.info(f"📄 Reading PDF: {pdf_path}")
    text_chunks = []

    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)

        # Read the rulebook page by page
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text = page.extract_text()

            if text:
                # FIX: PyPDF2 does not return double newlines. 
                # We use an overlapping sliding window of 150 words (safely under the 256 token limit).
                # The 30-word overlap ensures we never accidentally chop a rule in half.
                words = text.split()
                chunk_size = 150
                overlap = 30
                
                for i in range(0, len(words), chunk_size - overlap):
                    chunk_words = words[i:i + chunk_size]
                    if len(chunk_words) > 20:  # Skip tiny meaningless fragments
                        chunk_text = " ".join(chunk_words)
                        text_chunks.append({
                            "text": chunk_text,
                            "page": page_num + 1
                        })

    logger.success(f"✂️ Sliding Window Chunking Complete: Extracted {len(text_chunks)} safe chunks.")
    return text_chunks


def upload_to_qdrant(chunks):
    logger.info("🧠 Loading Sentence Transformer AI Model (this converts text to math)...")
    # This specific AI model outputs a 384-dimensional mathematical vector
    model = SentenceTransformer('all-MiniLM-L6-v2')

    logger.info("📡 Connecting to Qdrant Vector Database...")
    # Connect to the local Qdrant Docker container we built in Phase 0!
    client = QdrantClient(url="http://localhost:6333")

    collection_name = "fia_rulebook"

    # Recreate the collection so we can safely run this script multiple times
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    logger.info("🔢 Converting paragraphs to Vectors and uploading to Qdrant...")
    points = []
    for idx, chunk in enumerate(chunks):
        # The Magic: Convert the English paragraph into a 384-dimensional vector array
        vector = model.encode(chunk["text"]).tolist()

        # Create a Qdrant Data Point (Vector + The original text + The page number)
        point = PointStruct(
            id=idx,
            vector=vector,
            payload={
                "text": chunk["text"],
                "page": chunk["page"]
            }
        )
        points.append(point)

    # Upload all the vectors into the database
    client.upsert(collection_name=collection_name, points=points)
    logger.success(f"✅ Successfully embedded and stored {len(points)} rules in Qdrant!")


if __name__ == "__main__":
    # Fix PyCharm's Working Directory quirks by using the absolute project path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    pdf_path = os.path.join(project_root, "data", "fia_2024_sporting_regulations.pdf")

    if not os.path.exists(pdf_path):
        logger.error(f"❌ Could not find {pdf_path}. Check the data folder!")
    else:
        chunks = extract_text_from_pdf(pdf_path)
        upload_to_qdrant(chunks)

