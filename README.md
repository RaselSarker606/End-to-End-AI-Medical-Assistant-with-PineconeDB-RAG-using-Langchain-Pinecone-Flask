# 📄 End-to-End AI Medical Assistant with Pinecone Retrieval-Augmented Generation

A **Retrieval-Augmented Generation (RAG)** system that extracts knowledge from PDFs, embeds them into a vector database, and provides intelligent answers using Google’s **Gemini LLM**.  
This project demonstrates how to integrate **LangChain**, **Pinecone**, and **HuggingFace embeddings** inside a Flask-based interface styled with **HTML & CSS**.  

---

## 📖 Overview

This system allows you to:

- 📂 Load and process PDFs  
- ✂️ Split text into optimized chunks  
- 🔍 Generate vector embeddings with HuggingFace  
- 🗂️ Store & retrieve data efficiently with Pinecone  
- 🤖 Query with Gemini LLM using RAG workflow  
- 💻 Interact via a Flask web app with a clean HTML/CSS UI  

---

## ⚙️ Features

- **PDF Loader** → Extracts data from multiple PDFs  
- **Text Chunking** → Splits large files into contextual pieces  
- **Embeddings** → Uses HuggingFace MiniLM for 384-dim vectors  
- **Pinecone Vector Store** → Fast and scalable vector search  
- **Retriever + RAG** → Pulls top relevant chunks for accurate answers  
- **Flask Web App** → User-friendly front-end with HTML & CSS  

---

## 🛠️ Tech Stack

- **LangChain** → Framework for chaining LLM + retrievers  
- **Pinecone** → Vector DB for semantic search  
- **HuggingFace** → Sentence embeddings  
- **Google Gemini** → LLM for response generation  
- **Flask** → Backend server  
- **HTML5 + CSS3** → Front-end UI  

---
## 📊 Example Workflow
User: Summarize the medical case study.  
BOT: The document discusses treatment steps, safety measures, and clinical outcomes in concise detail.

---
## 📌 Future Improvements

🔐 Secure API keys with .env

🌐 Multi-file support (DOCX, TXT, JSON)

⚡ Async requests for speed

🎨 Advanced front-end UI with JS interactivity

☁️ Docker/Cloud deployment for scalability

🧠 Support for multilingual embeddings
---

## 🙋‍♂️ About Me

Hi, I’m Rasel Sarker 👋

🔍 Skilled in Pinecone Vector DB for semantic search

⚡ Experienced with RAG pipelines using LangChain

🤖 Build intelligent chatbots powered by embeddings + LLMs

💻 Develop Flask web apps with HTML & CSS frontends

🚀 Passionate about AI-powered applications and real-world NLP solutions

---
