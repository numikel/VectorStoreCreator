import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

class VectorStoreCreator:
    """
    Creates and loads a FAISS vectorstore from documents located in a directory.
    Supports multilingual embeddings and recursive character-based text splitting.
    """

    def __init__(
        self,
        knowledge_path: str,
        db_name: str,
        db_path: str = "vectorstores",
        chunk_size: int = 5000,
        chunk_overlap: int = 500,
        k: int = 5
    ):
        """
        :param knowledge_path: Path to directory containing source documents.
        :param db_name: Name of the FAISS vectorstore to create or load.
        :param db_path: Directory where vectorstores are saved.
        :param chunk_size: Maximum number of characters in a chunk.
        :param chunk_overlap: Overlap in characters between consecutive chunks.
        :param k: Number of top similar documents to retrieve.
        """
        os.makedirs(db_path, exist_ok=True)
        if not os.path.exists(knowledge_path):
            raise FileNotFoundError(f"Directory '{knowledge_path}' does not exist.")

        self.knowledge_path = knowledge_path
        self.db_path = db_path
        self.db_name = db_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k

        self.loader = DirectoryLoader(self.knowledge_path)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        self.embedding_model = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large-instruct"
        )

        self.vectorstore = self._create_vectorstore()
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": self.k}
        )

    def _load_documents(self):
        print("[RAGDocumentProcessor] Loading documents...")

        # Sprawdź, czy folder z dokumentami jest pusty
        if not os.listdir(self.knowledge_path):
            print(f"[RAGDocumentProcessor] WARNING: Folder '{self.knowledge_path}' is empty!")
            return []

        return self.loader.load()

    def _chunk_documents(self, documents):
        print("[RAGDocumentProcessor] Splitting documents into chunks...")
        return self.text_splitter.split_documents(documents)

    def _create_vectorstore(self):
        full_db_path = os.path.join(self.db_path, self.db_name)

        if not os.path.exists(full_db_path):
            print("[RAGDocumentProcessor] Creating new vectorstore...")
            documents = self._load_documents()

            if not documents:
                raise ValueError(f"No documents in '{self.knowledge_path}'. Folder is empty or documents could not be loaded.")

            documents = self._chunk_documents(documents)
            vectordb = FAISS.from_documents(documents, embedding=self.embedding_model)
            vectordb.save_local(full_db_path)
        else:
            print("[RAGDocumentProcessor] Loading existing vectorstore...")
            vectordb = FAISS.load_local(
                full_db_path,
                embeddings=self.embedding_model,
                allow_dangerous_deserialization=True  # required due to FAISS format
            )

        return vectordb
    
    def update_vectorstore(self, force_rebuild: bool = False):
        """
        Updates the existing vectorstore based on changes in the source documents.
        If force_rebuild is True, the vectorstore will be deleted and rebuilt from scratch.
        """

        full_db_path = os.path.join(self.db_path, self.db_name)

        if force_rebuild or not os.path.exists(full_db_path):
            print(f"[RAGDocumentProcessor] {'Rebuilding' if force_rebuild else 'Creating'} vectorstore...")
            documents = self._load_documents()
            documents = self._chunk_documents(documents)
            vectordb = FAISS.from_documents(documents, embedding=self.embedding_model)
            vectordb.save_local(full_db_path)
            self.vectorstore = vectordb
            self.retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": self.k})
            print("[RAGDocumentProcessor] Vectorstore updated successfully.")
        else:
            print("[RAGDocumentProcessor] No changes detected or rebuild not requested. Using existing vectorstore.")

    def load_vectordb(self):
        """Returns the loaded FAISS vectorstore object."""
        return self.vectorstore

    def load_retriever(self):
        """Returns the retriever interface for the vectorstore."""
        return self.retriever


if __name__ == "__main__":
    # Example use case: create vectorstores for multiple document sets with full parameter control
    db_names = ["zemsta"]

    for db_name in db_names:
        print(f"Initializing vectorstore and retriever for '{db_name}'...")
        creator = VectorStoreCreator(
            knowledge_path=f"docs/{db_name}",     # Path to source documents
            db_name=db_name,                      # Name of the FAISS vectorstore
            db_path="vectorstores",               # Where to save the vectorstore
            chunk_size=4000,                      # Max characters per chunk
            chunk_overlap=300,                    # Overlap between chunks
            k=3                                   # Number of similar documents to retrieve
        )
        print(f"Vectorstore and retriever initialized for '{db_name}'")
        print()