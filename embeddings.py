from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Create vector database
def create_vector_db():
    """ Loading the list of PDF in the data folder
    Creating chunks and storing as an embeddings in vector DB """
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                        model_kwargs={'device': 'cpu'})
    
    ####### To run the below Instruct Embeddings please install sentence-transfomer=2.2.2 #######
    # embeddings = HuggingFaceInstructEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
    #                                     model_kwargs={'device': 'cpu'})
    
    # embeddings = HuggingFaceEmbeddings(model_name='thenlper/gte-small',
    #                                     multi_process=True,
    #                                     model_kwargs={'device': 'cpu'},
    #                                     encode_kwargs={"normalize_embeddings": True},)
    
    ####### Uncomment the below vector_db creation command only if 'thenlper/gte-small' model is being used #########
    # vector_db = FAISS.from_documents(texts, embeddings, distance_strategy=DistanceStrategy.COSINE)
    
    vector_db = FAISS.from_documents(texts, embeddings)
    vector_db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()

