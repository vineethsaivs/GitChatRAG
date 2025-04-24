**GitChatRAG**  
A Streamlit‑powered offline Retrieval‑Augmented Generation (RAG) app to interactively query any public GitHub repository using a local Ollama LLM.

---

## 🚀 Features

- **Offline-first RAG**: Ingests a GitHub repo into plaintext chunks, builds a FAISS index for vector search, and answers questions locally.  
- **Ollama Integration**: Leverages any Ollama‑compatible LLM (e.g., `mistral`, `llama3`, `phi3`) for generation without external API calls.  
- **Streamlit UI**: Sleek sidebar controls for loading/re‑indexing repos and model selection, plus chat history with clear conversation threads.  
- **Chain‑of‑Thought Prompting**: Built‑in few‑shot examples and step‑by‑step reasoning instructions for more accurate code and documentation answers.  

---

## 📦 Getting Started

### Prerequisites

- Python 3.11+  
- [Ollama](https://ollama.com) installed and initialized locally  
- Git installed  

### Install dependencies

```bash
git clone https://github.com/vineethsaivs/GitChatRAG.git
cd GitChatRAG
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install ollama-python
```

### Pull your Ollama model

```bash
# Replace `mistral` with your preferred model name
ollama pull mistral
```

### Run the app

```bash
streamlit run app.py
```

---

## ⚙️ Configuration

- No environment variables are strictly required — `.env` support is built in for future secrets.  
- **Sidebar Inputs**:  
  - **GitHub URL**: Any public repo URL (e.g., `https://github.com/user/repo`).  
  - **Ollama model**: Name of your local Ollama model.  

---

## 🔍 How It Works

1. **Ingest & Index**  
   - Uses `gitingest.ingest(repo_url)` to clone the repo, extract text across files, and return a single concatenated string.  
   - Splits text into 600‑char chunks, embeds with `sentence-transformers/all-MiniLM-L6-v2`, and adds to a FAISS `IndexFlatL2`.

2. **Query & Retrieve**  
   - On each user query, embed the question, retrieve top‑5 nearest chunks from FAISS, and assemble context.  

3. **Generate Answer**  
   - Combines chain‑of‑thought system instructions, few‑shot examples, and repo context into a prompt.  
   - Calls `ollama.generate(...)` to stream back a concise, step‑by‑step answer, referencing file names when applicable.

4. **Chat UI**  
   - Maintains session‑state history (`st.session_state.messages`) for back‑and‑forth dialogue.  
   - Offers a clear “↺ Clear” button to reset conversation and reduce memory usage.

---

## 🛠️ Development

- **Embedder Cache**: `@st.cache_resource` for SentenceTransformer instantiation.  
- **Ollama Cache**: Pulls and caches your chosen Ollama model.  
- **Index Cache**: Persists FAISS index per repo key in `st.session_state.index_cache` for instant lookup.


### Directory Structure

```
├── app.py
├── requirements.txt
├── gitingest/       # Repo ingestion helper
├── models/          # (Optional) Custom embedding or LLM wrappers
└── README.md        # You are here!
```

---

## 🤝 Contributing

1. Fork the repo.  
2. Create a feature branch (`git checkout -b feature/my-feature`).  
3. Commit your changes (`git commit -m "Add my feature"`).  
4. Push to the branch (`git push origin feature/my-feature`).  
5. Open a Pull Request and describe your changes.

Please ensure your code passes linting and tests before submission.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*Happy local coding!* ❤️
