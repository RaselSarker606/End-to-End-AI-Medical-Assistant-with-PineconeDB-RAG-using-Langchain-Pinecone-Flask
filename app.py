from flask import Flask, render_template, jsonify, request
from prompt import *
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

from store_index import docsearch
from prompt import system_prompt

# Flask app setup =============================================================

app = Flask(__name__)

# Load the .env file
load_dotenv()

PINECONE_API_KEY=os.environ.get("PINECONE_API_KEY")
GOOGLE_API_KEY=os.environ.get("GOOGLE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7,
        convert_system_message_to_human=True,
        verbose=True
    )
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])


QA_chain = create_stuff_documents_chain(llm, prompt)
main_chain = create_retrieval_chain(retriever, QA_chain)

@app.route("/")
def index():
    return render_template("index.html")

# Response =================================================================
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = main_chain.invoke({"input": msg})
    print("Response : ", response["answer"] )
    return str(response["answer"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port= 8080, debug= True)
