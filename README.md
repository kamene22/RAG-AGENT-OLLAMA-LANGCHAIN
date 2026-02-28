# RAG Agent with Ollama and LangChain

A Retrieval-Augmented Generation (RAG) agent that answers questions about restaurant reviews using local LLMs via Ollama, vector embeddings, and LangChain.

## 🚀 Features

- **Local LLM Integration**: Uses Ollama to run language models locally (no API keys required)
- **Vector Search**: Leverages ChromaDB for efficient semantic search over restaurant reviews
- **RAG Pipeline**: Retrieves relevant context from reviews before generating answers
- **Interactive Q&A**: Command-line interface for asking questions about restaurant reviews
- **Persistent Storage**: Vector database persists between sessions for faster subsequent runs

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8+**
- **Ollama** - [Download and install Ollama](https://ollama.ai/)
- **Required Ollama Models**:
  - `llama3.2:3b` - For the language model
  - `mxbai-embed-large` - For embeddings

### Installing Ollama Models

After installing Ollama, pull the required models:

```bash
ollama pull llama3.2:3b
ollama pull mxbai-embed-large
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kamene22/RAG-AGENT-OLLAMA-LANGCHAIN.git
   cd RAG-AGENT-OLLAMA-LANGCHAIN
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage

### 1. Set Up the Vector Database

First, run the vector setup script to create embeddings from the restaurant reviews:

```bash
python vector.py
```

This will:
- Load restaurant reviews from `realistic_restaurant_reviews.csv`
- Generate embeddings using the `mxbai-embed-large` model
- Store them in ChromaDB at `./chroma_langchain_db`
- Create a retriever that fetches the top 5 most relevant reviews

**Note**: This only needs to be run once (or when you update the CSV file). The database persists between runs.

### 2. Run the RAG Agent

Start the interactive Q&A session:

```bash
python main.py
```

You'll be prompted to ask questions about the restaurant reviews. Type `q` to quit.

### Example Questions

- "What is the best pizza place in town?"
- "What are customers saying about the crust?"
- "Are there any complaints about delivery?"
- "What do people like about the service?"

## 📁 Project Structure

```
RAG-AGENT-OLLAMA-LANGCHAIN/
│
├── main.py                          # Main RAG agent with interactive Q&A
├── vector.py                        # Vector database setup and retriever
├── realistic_restaurant_reviews.csv # Restaurant review dataset
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore file
├── README.md                        # This file
└── chroma_langchain_db/             # ChromaDB storage (created after first run)
```

## 🔧 How It Works

1. **Vector Database Setup** (`vector.py`):
   - Loads restaurant reviews from CSV
   - Creates embeddings for each review using Ollama
   - Stores embeddings in ChromaDB with metadata (rating, date)
   - Creates a retriever that returns top-k relevant documents

2. **RAG Pipeline** (`main.py`):
   - Takes user question
   - Retrieves top 5 most relevant reviews using semantic search
   - Passes retrieved context and question to LLM
   - Generates answer based on the retrieved reviews

## 🧪 Technologies Used

- **[LangChain](https://www.langchain.com/)** - Framework for building LLM applications
- **[Ollama](https://ollama.ai/)** - Local LLM runtime
- **[ChromaDB](https://www.trychroma.com/)** - Vector database for embeddings
- **[Pandas](https://pandas.pydata.org/)** - Data manipulation and CSV handling

## ⚙️ Configuration

### Changing the LLM Model

Edit `main.py` to use a different Ollama model:

```python
model = ChatOllama(model="your-model-name")
```

### Changing the Embedding Model

Edit `vector.py` to use a different embedding model:

```python
embeddings = OllamaEmbeddings(model="your-embedding-model")
```

### Adjusting Retrieval Parameters

Modify the number of retrieved documents in `vector.py`:

```python
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}  # Change 5 to desired number
)
```

## 📝 Dataset

The project includes `realistic_restaurant_reviews.csv` with restaurant reviews containing:
- **Title**: Review title
- **Date**: Review date
- **Rating**: Rating (1-5)
- **Review**: Full review text

You can replace this with your own CSV file, ensuring it has at least a "Review" column (or modify `vector.py` to match your schema).

## 🐛 Troubleshooting

### Ollama Connection Issues

- Ensure Ollama is running: `ollama serve`
- Verify models are installed: `ollama list`
- Check if models are accessible: `ollama show llama3.2:3b`

### ChromaDB Errors

- Delete `chroma_langchain_db/` folder and rerun `vector.py` to recreate the database
- Ensure you have write permissions in the project directory

### Import Errors

- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Verify you're using the correct Python environment

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [LangChain](https://www.langchain.com/) for the excellent framework
- [Ollama](https://ollama.ai/) for making local LLMs accessible
- [ChromaDB](https://www.trychroma.com/) for vector database capabilities

---

**Note**: Make sure Ollama is running before executing the scripts. The first run may take longer as it generates embeddings for all reviews.
