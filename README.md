# 📘 VectorStoreCreator – Vectorstore Generator for RAG Pipelines

This project enables fast and scalable creation of **FAISS vectorstores** from various document types using LangChain, multilingual embeddings, and advanced recursive text splitting.  
It is designed for use in **RAG (Retrieval-Augmented Generation)** systems, such as LLM-based question answering, document assistants, or knowledge-grounded chatbots.

---

## 📂 Project Structure
```bash
VectorStoreCreator/
├── vectorstores_creator.py      # Core logic for processing, embedding, saving
├── requirements.txt             # Python dependencies
├── .env                         # Optional environment variables
├── LICENSE                      # Project license (MIT)
└── README.md                    # Project documentation
```

---

## ⚙️ Features

- 🔗 LangChain-compatible architecture  
- 🧠 Multilingual embeddings with `intfloat/multilingual-e5-large-instruct`  
- ✂️ Recursive chunking with overlap support  
- 📦 FAISS vectorstore creation and persistence  
- 🔎 Built-in retriever interface  
- 📄 Supports multiple document types: `.txt`, `.pdf`, `.docx`, `.md`, etc.

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add documents
Place your documents (e.g. .txt, .pdf, .docx) in folders inside the docs/ directory.
Each folder will be processed as a separate vectorstore.

Example:

```bash
docs/pan_tadeusz/pan_tadeusz.txt
docs/zemsta/zemsta.pdf
```

### 3. Run the script
```bash
python vectorstores_creator.py
```

This will create a FAISS vectorstore for each folder defined in the db_names list in vectorstores_creator.py.

---

## 💡 Example usage (Python)
```python
from vectorstores_creator import VectorStoreCreator

creator = VectorStoreCreator(
    knowledge_path="docs/zemsta",
    db_name="zemsta",
    db_path="vectorstores",
    chunk_size=4000,
    chunk_overlap=300,
    k=3
)

vectorstore = creator.load_vectordb()
retriever = creator.load_retriever()
```

---

## 🔄 Updating a vectorstore
If you make changes to your source documents and want to regenerate a vectorstore, you can use the built-in .update_vectorstore() method.

```python
creator = VectorStoreCreator(
    knowledge_path="docs/zemsta",
    db_name="zemsta",
    db_path="vectorstores",
    chunk_size=4000,
    chunk_overlap=300,
    k=3
)

# Option 1: Only rebuild if the vectorstore does not exist
creator.update_vectorstore()

# Option 2: Force rebuild, even if the vectorstore exists
creator.update_vectorstore(force_rebuild=True)
```

This approach is useful when:
1) you've added or modified documents in the source folder,
2) you want to refresh the embeddings without manually deleting the FAISS index.

---

## ✅ Supported file types
This project supports most document formats via LangChain loaders:

.txt, .pdf, .docx, .md, .html, .csv, .pptx, and more

Format support depends on your system and installed packages

---

## 🔧 Optional system dependencies
For best file type support:

### Windows
```bash
pip install python-magic-bin
```

### Linux/macOS
```bash
sudo apt install libmagic1
pip install python-magic
```

---

## 👤 Author
Made with 🧠 by Michał Kamiński

---

## 📄 License
This project is licensed under the MIT License. You are free to use, modify, and distribute it as you wish.
