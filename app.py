# app.py  –  chat with any public GitHub repo, completely offline via Ollama
import os, gc, uuid, logging, textwrap
from typing import Tuple, List

import streamlit as st
from dotenv import load_dotenv
from gitingest import ingest                    # repo → plaintext dump

from sentence_transformers import SentenceTransformer
import faiss, numpy as np
import ollama                                   # >>> pip install ollama-python


# ── basic setup ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()                                   # (spare, for future secrets)

if "id" not in st.session_state:
    st.session_state.id          = uuid.uuid4().hex
    st.session_state.index_cache = {}           # repo-key → (faiss, chunks)
    st.session_state.messages    = []           # chat history

# ── helpers ────────────────────────────────────────────────────────────────
def repo_key(url: str) -> str:
    name = url.rstrip("/").split("/")[-1].replace(".git", "")
    return f"{st.session_state.id}-{name}"

def valid_url(u: str) -> bool:
    return u.startswith(("https://github.com/", "http://github.com/"))

def reset_chat():
    st.session_state.messages.clear()
    gc.collect()

# ── cached models ──────────────────────────────────────────────────────────
@st.cache_resource
def embedder() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")      # ~100 MB, CPU-only, free

@st.cache_resource
def ollama_model(name: str):
    """Returns a tiny wrapper that calls the chosen Ollama model."""
    try:
        ollama.pull(name)
    except Exception as e:
        st.sidebar.error(f"Ollama pull failed: {e}")
        raise
    return name

def llama_reply(model: str, prompt: str) -> str:
    """Sync call to Ollama; returns the full response text."""
    resp = ollama.generate(
        model=model,
        prompt=prompt,
        stream=False,
        options={
            "temperature": 0.7,
            "top_p": 0.9,
            "repeat_penalty": 1.15,
            "num_predict": 256
        }
    )
    return resp["response"].strip()

# ── vector-store helpers ───────────────────────────────────────────────────
def chunk(txt: str, *, max_chars: int = 600) -> List[str]:
    return [
        txt[i : i + max_chars].strip()
        for i in range(0, len(txt), max_chars)
        if txt[i : i + max_chars].strip()
    ]

def build_index(long_text: str) -> Tuple[faiss.IndexFlatL2, List[str]]:
    pieces = chunk(long_text)
    embeds = embedder().encode(pieces, convert_to_numpy=True, show_progress_bar=False)
    index  = faiss.IndexFlatL2(embeds.shape[1])
    index.add(embeds)
    return index, pieces

# ── sidebar – repo & model controls ────────────────────────────────────────
st.sidebar.header("📂 Load GitHub Repo")
gh_url = st.sidebar.text_input("GitHub URL", placeholder="https://github.com/user/repo")

st.sidebar.markdown("---")
ollama_choice = st.sidebar.text_input("🦙 Ollama model", value="mistral")  # e.g. llama3, mistral, phi3

if st.sidebar.button("Load / Re-index"):
    try:
        if not valid_url(gh_url):
            st.sidebar.error("Invalid GitHub URL"); st.stop()

        key = repo_key(gh_url)
        with st.spinner("Ingesting & indexing…"):
            _, _, txt = ingest(gh_url)
            if not txt:
                raise RuntimeError("Ingest returned empty text.")
            st.session_state.index_cache[key] = build_index(txt)

        # (re-)initialise/confirm model
        ollama_model(ollama_choice)

        reset_chat()
        st.sidebar.success("✅ Ready to chat!")
    except Exception as e:
        st.sidebar.error(e)
        logger.exception("load-error")

# ── main chat UI ───────────────────────────────────────────────────────────
col1, col2 = st.columns([6, 1])
col1.header("💬 Chat with Repo")
col2.button("↺ Clear", on_click=reset_chat)

for m in st.session_state.messages:  # chat history
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if (q := st.chat_input("Ask anything…")):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("assistant"):
        box = st.empty()
        try:
            idx, chunks = st.session_state.index_cache[repo_key(gh_url)]

            # embed & search
            q_vec = embedder().encode([q], convert_to_numpy=True)
            _, I  = idx.search(q_vec, 5)  # top-5 passages
            ctx   = "\n\n---\n\n".join(chunks[i] for i in I[0])

            # ── Few-shot examples ────────────────────────────────────────
            few_shot = textwrap.dedent("""
            Example 1:
            Context:
              File structure:
                - main.py: loads data and runs training loop
                - utils.py: data normalization functions
            Question:
              How does the data normalization work?
            Answer:
              1. Locate `normalize()` in `utils.py`.
              2. It subtracts the mean and divides by standard deviation.
              3. Called in `main.py` before each training batch.
              **Final Answer**: Data normalization is implemented in `utils.py`’s `normalize()` which standardizes inputs (mean zero, unit variance) and is invoked in `main.py`.

            Example 2:
            Context:
              File structure:
                - api.py: defines FastAPI routes
                - models.py: Pydantic schemas `User`, `Item`
            Question:
              Which endpoint returns all items for a user?
            Answer:
              1. In `api.py`, inspect `/users/{user_id}/items`.
              2. This route calls `get_items_by_user` in `models.py`.
              3. It returns a list of `Item` schemas.
              **Final Answer**: The endpoint `GET /users/{user_id}/items` in `api.py` returns all items for that user.
            """).strip()

            # ── Chain-of-Thought System Instructions ─────────────────────
            system_instructions = textwrap.dedent("""
            You are a highly-capable coding assistant.
            When you answer:
              • First, think *step by step* (chain-of-thought) and outline your reasoning  
              • Then, provide a concise final answer under **“Final Answer:”**  
              • Always reference file names and paths when relevant  
              • If you don’t know, say “I don’t know.”
            """).strip()

            # ── Assemble full prompt ────────────────────────────────────
            prompt = "\n\n".join([
                system_instructions,
                few_shot,
                "### Repository Context\n" + ctx,
                "### User Question\n" + q,
                "### Your Answer"
            ])

            answer = llama_reply(ollama_choice, prompt)
            box.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except KeyError:
            box.markdown("⚠️ Load a repository first!")
        except Exception as e:
            box.markdown("🙁 Sorry, something went wrong.")
            logger.exception("chat-error")
