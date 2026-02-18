# GDG Knowledge Agent

An intelligent RAG (Retrieval-Augmented Generation) application built with Google's Gemini AI and Streamlit. This agent helps users find information from documentation, guidelines, and even live websites using semantic search and a vector database.

## Features

- **RAG Architecture**: Combines vector search (ChromaDB) with LLM generation (Gemini) for accurate, sourced answers.
- **Knowledge Base**: 
  - **Auto-ingestion**: Automatically loads local text files.
  - **File Upload**: Users can upload `.txt` or `.md` files to expand the knowledge base.
  - **Web Scraping**: Fetch and process content from live URLs.
- **Interactive UI**: Built with Streamlit for a clean, chat-like experience.
- **Source Attribution**: Every answer includes citations from the source documents.

## Prerequisites

- Python 3.8+
- A Google Cloud Project with Gemini API access
- An API Key from [Google AI Studio](https://aistudio.google.com/app/apikey)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables**:
   Create a `.env` file in the root directory and add your API Key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

### Run the Web App
Start the Streamlit application:
```bash
streamlit run streamlit_app.py
```
The app will open in your default browser at `http://localhost:8501`.

### Run via Command Line
You can also run the agent in the terminal for testing:
```bash
python rag_agent.py
```

## Project Structure

- `streamlit_app.py`: Main web application interface.
- `rag_agent.py`: Core RAG agent logic (Retrieval + Generation).
- `knowledge_base.py`: Vector database management using ChromaDB.
- `gemini_wrapper.py`: Wrapper for Google's Gemini API.
- `chunking_utility.py`: Text processing and intelligent chunking.
- `text_cleaner.py`: Text normalization utilities.
- `semantic_similarity.py`: Vectors and similarity calculation helpers.
- `faq_finder.py`: specialized FAQ matching logic.

## License

[MIT](LICENSE)
