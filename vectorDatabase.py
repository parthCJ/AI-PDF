from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

pdfsDirectory = 'pdfs/'

def uploadPDF(file):
    with open(pdfsDirectory + file.name, "wb") as f:
        f.write(file.getbuffer())
 
def loadPDF(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

# file_path = "pdf's/Divyansh20rathore.pdf"
file_path = "uploaded_pdfs/current_document.pdf"
documents = loadPDF(file_path)
print(len(documents))

def create_chunks(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

text_chunks = create_chunks(documents)
print(("Chunks Count: "), len(text_chunks))

ollama_model_name = "deepseek-r1:7b"
def get_embeddings_model(ollama_model_name):
    embeddings = OllamaEmbeddings(model=ollama_model_name)
    return embeddings

FAISS_DB_PATH = "vectorstore/db_faiss"
faiss_db = FAISS.from_documents(text_chunks, get_embeddings_model(ollama_model_name))
faiss_db.save_local(FAISS_DB_PATH)

