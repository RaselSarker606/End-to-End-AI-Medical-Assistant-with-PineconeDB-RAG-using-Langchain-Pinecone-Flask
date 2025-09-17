'''from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


# Extract Data From the PDF File ========================================================
def load_pdf_file(data):
    loader = DirectoryLoader(data,
                             glob=".pdf",
                             loader_class=PyPDFLoader)
    documents = loader.load()
    return documents


# Split the Data into Text Chunks ===============================================================
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


# Download the Embedding Model from Hugging Face =============================================================
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')#384 dimensional
    return embeddings


# Vector Database using Pinecone ==========================================================================
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

pc = Pinecone(api_key="Your API Key")

index_name = "medicaldocument"

pc.create_index(
    name=index_name,
    dimension=384,
    metric='cosine',
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1",
    )
)


# Convert to Embedding each chunk and Load Existing Index in Pinecone VectorStore  ========================================================================
from langchain_pinecone import PineconeVectorStore

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)
# Load Existing Index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

# Retriever ======================================================================
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


# Large Language Model (LLM) Implementation =========================================================================================================
from langchain_google_genai import ChatGoogleGenerativeAI

GOOGLE_API_KEY = "Your API Key"

llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7,
        convert_system_message_to_human=True,
        verbose=True
    )


# Prompt Template Style ========================================================================================
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "You are a helpful assistant for answering questions using the provided context only. "
    "Read the retrieved context carefully and use it to answer the question accurately. "
    "If the answer is not present in the context, respond with 'I don't know.' "
    "Keep your response concise, clear, and no longer than three sentences."
    "\n\n"

    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])



# Make a Chain =============================================================================================
QA_chain = create_stuff_documents_chain(llm, prompt)
main_chain = create_retrieval_chain(retriever, QA_chain)

response = main_chain.invoke({input})'''








