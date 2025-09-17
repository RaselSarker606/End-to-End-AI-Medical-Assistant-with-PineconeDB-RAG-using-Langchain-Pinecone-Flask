# store_index.py

from helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
import os
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicaldocument"

# Download embeddings model once
embeddings = download_hugging_face_embeddings()

# Check if index exists
existing_indexes = [idx.name for idx in pc.list_indexes()]
if index_name not in existing_indexes:
    print(f"[INFO] Creating new Pinecone index: {index_name}")

    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        )
    )

    # Load and process PDF data only on first creation
    extracted_data = load_pdf_file(data='Books/')
    text_chunks = text_split(extracted_data)

    # Upload to Pinecone
    docsearch = PineconeVectorStore.from_documents(
        documents=text_chunks,
        index_name=index_name,
        embedding=embeddings,
    )
    print("[INFO] Data uploaded to Pinecone index successfully.")
else:
    print(f"[INFO] Loading existing Pinecone index: {index_name}")
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
    )
    print("[INFO] Existing index loaded successfully.")
