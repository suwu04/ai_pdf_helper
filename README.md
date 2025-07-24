# Chat with Multiple PDFs

An interactive chatbot app that allows users to upload PDF documents and ask natural language questions about their contents. Built using **LangChain**, **Streamlit**, **FAISS**, and either **OpenAI** or **HuggingFace** LLMs.

---

## Features

*  Upload and process multiple PDF files
*  Automatic text extraction and intelligent chunking
*  Semantic search using vector embeddings (OpenAI or HuggingFace Instructor)
*  Conversational memory using LangChain’s `ConversationBufferMemory`
*  Graceful handling of non-PDF and corrupt files
*  Streamlit-powered chat interface with avatars
*  Secure environment configuration via `.env`

---

## Tech Stack

| Component        | Technology                                       |
| ---------------- | ------------------------------------------------ |
| **Frontend**     | Streamlit                                        |
| **PDF Parsing**  | PyPDF2                                           |
| **Embeddings**   | OpenAIEmbeddings / HuggingFaceInstructEmbeddings |
| **Vector Store** | FAISS                                            |
| **LLM**          | ChatOpenAI or HuggingFaceHub (e.g., FLAN-T5)     |
| **Memory**       | LangChain `ConversationBufferMemory`             |
| **Styling**      | Custom HTML/CSS templates with avatars           |
| **Config**       | `python-dotenv` for secure environment variables |

---
## Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/pdf-chatbot.git
cd pdf-chatbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note**: To use HuggingFace's instructor embeddings, uncomment the related lines in `requirements.txt`.

### 3. Set up Environment Variables

Create a `.env` file in the root directory and add your API keys:

```
OPENAI_API_KEY=your_openai_key_here
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

### 4. Run the App

```bash
streamlit run app.py
```

---

## Switching Between OpenAI and HuggingFace

* **Embeddings**: Replace `OpenAIEmbeddings()` with `HuggingFaceInstructEmbeddings(...)` in `get_vectorstore()`
* **LLM**: Replace `ChatOpenAI()` with `HuggingFaceHub(...)` in `get_conversation_chain()`

 This app defaults to HuggingFace models to reduce API costs.

---

## Known Issues

*  API endpoint formatting can cause occasional errors during switching
*  Still exploring truly **free** alternatives for LLMs + embedding APIs
*  HuggingFace Instructor model is **resource-intensive** (best run on a GPU machine)

---

## Future Improvements

* Add offline embedding support with sentence-transformers
* Integrate ChromaDB as a vector DB alternative
* Enhance avatar-based UI customization

---

## License

MIT License
