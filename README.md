# Heritage-RAG: AI-Powered Heritage Conservation Assistant

## 📌 Project Overview

Heritage-RAG is a Retrieval-Augmented Generation (RAG) application designed to help users explore heritage conservation documents and monument-related information. The application allows users to upload PDF documents, retrieve relevant information, and ask questions in natural language. It generates answers based on the uploaded documents and displays source references to support the responses.

## 🎯 Objectives

- Simplify access to heritage and monument conservation information.
- Extract relevant information from PDF documents.
- Provide question answering using Retrieval-Augmented Generation.
- Display source references to help users verify answers.
- Demonstrate the application of AI and semantic search in cultural heritage research.

## ✨ Features

- **PDF Upload:** Upload text-based PDF documents.
- **Text Extraction:** Extract document text for processing.
- **Semantic Search:** Retrieve relevant passages using sentence embeddings.
- **AI Question Answering:** Generate answers using a language model.
- **Source References:** Display supporting document sources and page information.
- **Web Deployment:** Access the application through Streamlit Community Cloud.

## 🛠️ Technologies Used

- **Python** — Core programming language
- **Streamlit** — Web application interface
- **PyMuPDF** — PDF text extraction
- **ChromaDB** — Vector database for document retrieval
- **Sentence Transformers** — Text embeddings for semantic search
- **Hugging Face Transformers** — Language model integration
- **Qwen2.5-0.5B-Instruct** — Text generation model
- **PyTorch** — Machine learning framework
- **GitHub** — Source code management
- **Streamlit Community Cloud** — Application hosting

## ⚙️ How It Works

1. The user uploads a heritage-related PDF document.
2. The application extracts text from the PDF.
3. The extracted text is divided into smaller passages.
4. Sentence Transformers converts the passages into vector embeddings.
5. ChromaDB stores the embeddings for similarity-based retrieval.
6. When the user asks a question, the application retrieves relevant passages.
7. The language model generates an answer using the retrieved context.
8. The application displays the answer along with source references.

## 🚀 Run Locally

### Prerequisites

- Python installed
- Git (optional, for cloning the repository)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR-USERNAME/heritage-rag.git
   ```

2. Navigate to the project directory:

   ```bash
   cd heritage-rag
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the application:

   ```bash
   streamlit run streamlit_app.py
   ```

5. Open the local URL displayed in your terminal.

## 🌐 Live Demo

**Streamlit App:** https://heritage-rag-new-dimqibqpckcslaxf34tbpm.streamlit.app/
## 📂 Project Structure

```text
heritage-rag/
├── streamlit_app.py    # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .gitignore          # Files excluded from Git
└── README.md           # Project documentation
```

## 🔮 Future Enhancements

- Support for multiple PDF documents.
- OCR support for scanned documents.
- Improved retrieval accuracy and answer evaluation.
- Enhanced document filtering and source navigation.
- Persistent storage for uploaded documents.

## 🎓 Conclusion

Heritage-RAG demonstrates how Retrieval-Augmented Generation can be used to make heritage conservation information easier to access. By combining PDF text extraction, semantic retrieval, and language model-based question answering, the application helps users find relevant information from heritage documents with supporting source references.
