# Intelligent PDF Reader

## Overview
Developed an intelligent PDF Reader that extracts content from PDFs and answers user queries based on the document. The system utilizes advanced natural language processing techniques to ensure accurate and efficient document comprehension and interaction.

## Features
- Extracts content from uploaded PDFs
- Answers user queries based on document content
- Implements a Retrieval-Augmented Generation (RAG) pipeline
- Provides an intuitive UI for seamless interaction
- Efficient vector storage for fast retrieval
- Scalable processing for handling large documents

## Technologies Used
- **Frontend/UI:** Streamlit
- **Framework:** LangChain
- **Embeddings:** Ollama
- **Vector Storage:** FAISS
- **LLM Processing:** DeepSeek R1 on Groq Cloud
- **Programming Language:** Python

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip (Python package manager)

### Steps
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/intelligent-pdf-reader.git
   cd intelligent-pdf-reader
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the application:
   ```sh
   streamlit run app.py
   ```

## Usage
1. Upload a PDF document via the UI.
2. Enter your query in the input field.
3. The system retrieves relevant information and generates a response.
4. View the extracted information and interact with the document efficiently.

## Contributing
Feel free to submit issues or pull requests to improve the project.
