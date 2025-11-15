# rag.py - Retrieval layer (PDF + TXT), FAISS/BM25 with simple domain filtering
from __future__ import annotations
import os, glob, re, time
from typing import Any, Dict, List, Optional, Tuple
from config import RAG_DOC_DIR, FAISS_DIR, NUMPY_AVAILABLE

# LangChain bits
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def _infer_category_from_path(path: str) -> str:
    """
    Very lightweight doc categorization based on filename.
    Examples:
      - 'mowing_standard.txt' -> 'mowing'
      - 'sport_field_standard.pdf' -> 'field'
    """
    n = os.path.basename(path).lower()
    if any(x in n for x in ["activity", "frequency", "maint", "maintenance"]):
        return "activity"
    if any(x in n for x in ["mowing", "turf", "grass", "lawn"]):
        return "mowing"
    if any(x in n for x in ["sport", "field", "soccer", "baseball", "softball", "rugby", "football"]):
        return "field"
    return "generic"


def _latest_mtime(paths: List[str]) -> float:
    """Return latest mtime across files, or 0.0 if none."""
    mt = 0.0
    for p in paths:
        try:
            mt = max(mt, os.path.getmtime(p))
        except Exception:
            pass
    return mt


class RAGIndex:
    """
    Primary: FAISS + HF embeddings (if numpy exists)
    Fallback: BM25 (no embeddings needed)
    Supports: PDF and TXT files
    Adds: simple category tagging + filter support; auto-rebuild FAISS when docs change
    """
    def __init__(self, doc_dir: str, faiss_dir: str):
        self.doc_dir = doc_dir
        self.faiss_dir = faiss_dir
        self.mode: str = "none"   # "faiss" | "bm25" | "none"
        self.emb = None
        self.vs: Optional[FAISS] = None
        self.retriever = None
        self._faiss_built_mtime: float = 0.0
        self._ensure_index()

    # --------------------------
    # Load & split documents
    # --------------------------
    def _load_docs(self) -> List[Document]:
        """Load PDFs and TXTs; attach metadata: source, page, category."""
        docs: List[Document] = []

        # TXT first (you replaced mowing_standard.pdf -> mowing_standard.txt)
        txts = sorted(glob.glob(os.path.join(self.doc_dir, "*.txt")))
        for t in txts:
            try:
                loaded = TextLoader(t, encoding="utf-8", autodetect_encoding=True).load()
                cat = _infer_category_from_path(t)
                for d in loaded:
                    meta = dict(d.metadata or {})
                    meta.setdefault("source", t)
                    meta.setdefault("page", 0)
                    meta["category"] = cat
                    docs.append(Document(page_content=d.page_content, metadata=meta))
                print(f"[RAG] Loaded TXT: {os.path.basename(t)} ({len(loaded)} docs) [category={cat}]")
            except Exception as e:
                print(f"[RAG] Failed to load TXT {t}: {e}")

        # PDFs (still supported if present)
        pdfs = sorted(glob.glob(os.path.join(self.doc_dir, "*.pdf")))
        for p in pdfs:
            try:
                loaded = PyPDFLoader(p).load()
                cat = _infer_category_from_path(p)
                for d in loaded:
                    meta = dict(d.metadata or {})
                    meta.setdefault("source", p)
                    # PyPDFLoader provides "page" in metadata; normalize to int
                    if "page" in meta and not isinstance(meta["page"], int):
                        try:
                            meta["page"] = int(meta["page"])
                        except Exception:
                            meta["page"] = 0
                    meta["category"] = cat
                    docs.append(Document(page_content=d.page_content, metadata=meta))
                print(f"[RAG] Loaded PDF: {os.path.basename(p)} ({len(loaded)} pages) [category={cat}]")
            except Exception as e:
                print(f"[RAG] Failed to load PDF {p}: {e}")

        print(f"[RAG] Total raw documents loaded: {len(docs)}")
        return docs

    def _split(self, docs: List[Document]) -> List[Document]:
        """
        Chunk documents for retrieval.
        TXT files can be long; we split everything uniformly.
        """
        if not docs:
            return []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=150,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        # Ensure all chunks keep category/source/page metadata
        for d in chunks:
            meta = d.metadata or {}
            meta.setdefault("source", "unknown")
            meta.setdefault("page", meta.get("page", 0))
            meta.setdefault("category", meta.get("category", _infer_category_from_path(meta.get("source", ""))))
            d.metadata = meta
        return chunks

    # --------------------------
    # Index build & refresh
    # --------------------------
    def _faiss_stamp_path(self) -> str:
        return os.path.join(self.faiss_dir, ".source_mtime")

    def _write_faiss_stamp(self, mtime: float) -> None:
        try:
            os.makedirs(self.faiss_dir, exist_ok=True)
            with open(self._faiss_stamp_path(), "w") as f:
                f.write(str(mtime))
        except Exception:
            pass

    def _read_faiss_stamp(self) -> float:
        try:
            with open(self._faiss_stamp_path(), "r") as f:
                return float(f.read().strip())
        except Exception:
            return 0.0

    def _ensure_index(self):
        docs = self._load_docs()
        if not docs:
            self.mode = "none"
            print("[RAG] No documents found in", self.doc_dir)
            return

        # Gather source list and latest mtime
        sources = sorted(set(d.metadata.get("source", "") for d in docs if d.metadata))
        latest_src_mtime = _latest_mtime(sources)

        if NUMPY_AVAILABLE:
            try:
                self.emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

                need_rebuild = True
                if os.path.isdir(self.faiss_dir) and os.listdir(self.faiss_dir):
                    # Compare stored mtime with current source mtime
                    stored = self._read_faiss_stamp()
                    if stored >= latest_src_mtime:
                        self.vs = FAISS.load_local(
                            self.faiss_dir, self.emb, allow_dangerous_deserialization=True
                        )
                        self.mode = "faiss"
                        self._faiss_built_mtime = stored
                        print(f"[RAG] Loaded existing FAISS index (up-to-date)")
                        return
                    else:
                        print(f"[RAG] Detected newer source files → rebuilding FAISS")

                # Build (or rebuild) FAISS
                chunks = self._split(docs)
                print(f"[RAG] Created {len(chunks)} chunks from {len(docs)} documents")

                self.vs = FAISS.from_documents(chunks, self.emb)
                os.makedirs(self.faiss_dir, exist_ok=True)
                self.vs.save_local(self.faiss_dir)
                self._write_faiss_stamp(latest_src_mtime)
                self._faiss_built_mtime = latest_src_mtime

                self.mode = "faiss"
                print(f"[RAG] Built and saved FAISS index")
                return
            except Exception as e:
                print(f"[RAG] FAISS failed → BM25 fallback: {e}")

        # BM25 fallback
        try:
            chunks = self._split(docs)
            self.retriever = BM25Retriever.from_documents(chunks)
            self.retriever.k = 5
            self.mode = "bm25"
            print(f"[RAG] Using BM25 retriever with {len(chunks)} chunks")
        except Exception as e:
            print(f"[RAG] BM25 failed: {e}")
            self.mode = "none"

    # --------------------------
    # Retrieval
    # --------------------------
    def _apply_filters(self, docs: List[Tuple[Document, float]] | List[Document], 
                       filters: Optional[dict], 
                       k: int) -> List[Tuple[Document, float]]:
        """
        Filter by simple criteria:
          filters = {
            "category": "mowing" | "field" | ...,
            "include_sources": ["mowing_standard.txt"],
            "exclude_sources": ["sport_field_standard.pdf"]
          }
        """
        if not docs:
            return []

        def doc_and_score(x):
            # normalize to (Document, score) for both FAISS and BM25 flows
            if isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], Document):
                return x
            return (x, 1.0)  # BM25 list result

        pairs = [doc_and_score(x) for x in docs]

        if not filters:
            return pairs[:k]

        cat = (filters.get("category") or "").strip().lower()
        inc = [s.lower() for s in (filters.get("include_sources") or [])]
        exc = [s.lower() for s in (filters.get("exclude_sources") or [])]

        def ok(d: Document) -> bool:
            meta = d.metadata or {}
            if cat and str(meta.get("category", "")).lower() != cat:
                return False
            src = os.path.basename(str(meta.get("source", ""))).lower()
            if inc and src not in inc:
                return False
            if exc and src in exc:
                return False
            return True

        filtered = [(d, s) for (d, s) in pairs if ok(d)]
        return filtered[:k]

    def retrieve(self, query: str, k: int = 4, filters: Optional[dict] = None) -> List[Dict[str, Any]]:
        hits: List[Dict[str, Any]] = []
        if not query:
            return hits

        if self.mode == "faiss" and self.vs is not None:
            raw = self.vs.similarity_search_with_score(query, k=max(k, 8))  # get a few extra, then filter
            docs = self._apply_filters(raw, filters, k)
            for i, (d, score) in enumerate(docs):
                meta = d.metadata or {}
                hits.append({
                    "doc_id": os.path.basename(meta.get("source", "unknown")),
                    "chunk_id": f"{i}",
                    "text": d.page_content,
                    "page": int(meta.get("page", 0)) + 1 if "page" in meta else None,
                    "image_ref": None,
                    "score": float(score),
                    "category": meta.get("category", "generic"),
                    "source": f"{meta.get('source','')}#p{int(meta.get('page',0))+1 if 'page' in meta else ''}",
                })
            return hits

        if self.mode == "bm25" and self.retriever is not None:
            raw_docs = self.retriever.get_relevant_documents(query)
            # Apply filters on the full set then truncate
            docs = [d for (d, _) in self._apply_filters(raw_docs, filters, k)]
            for i, d in enumerate(docs[:k]):
                meta = d.metadata or {}
                hits.append({
                    "doc_id": os.path.basename(meta.get("source", "unknown")),
                    "chunk_id": f"{i}",
                    "text": d.page_content,
                    "page": int(meta.get("page", 0)) + 1 if "page" in meta else None,
                    "image_ref": None,
                    "score": float(1.0 / (i + 1)),
                    "category": meta.get("category", "generic"),
                    "source": f"{meta.get('source','')}#p{int(meta.get('page',0))+1 if 'page' in meta else ''}",
                })
            return hits

        return hits


# --------------------------
# Global index + public API
# --------------------------
RAG = RAGIndex(RAG_DOC_DIR, FAISS_DIR)

def kb_retrieve(query: str, top_k: int = 3, filters: Optional[dict] = None):
    """
    Retrieve knowledge base document snippets.

    filters example:
      {"category": "mowing"}                          -> only mowing docs
      {"include_sources": ["mowing_standard.txt"]}    -> only that file
      {"exclude_sources": ["sport_field_standard.pdf"]}
    """
    hits = RAG.retrieve(query, k=top_k, filters=filters)
    return {"hits": hits}

def sop_extract(snippets: List[str], schema: Optional[List[str]] = None):
    """
    Extract structured SOP information from document snippets.

    Args:
        snippets: List of document text snippets
        schema: Optional extraction schema (reserved for future use)
    """
    if not snippets:
        return {"steps": [], "materials": [], "tools": [], "safety": []}

    text = "\n".join(snippets)
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    def pick(pred):
        return [l for l in lines if pred(l)]

    # Steps: numbered/bulleted lines
    steps = [l for l in lines if re.match(r"^(\d+[\).\s]|•|-)\s", l)]

    # Materials
    mats = pick(lambda l: re.search(
        r"(material|fertilizer|seed|mulch|line|marking|fuel)", l, re.I
    ))

    # Tools
    tools = pick(lambda l: re.search(
        r"(mower|edger|trimmer|blower|truck|line marker|roller|equipment)", l, re.I
    ))

    # Safety
    safety = pick(lambda l: re.search(
        r"(safety|PPE|goggles|hearing|lockout|traffic|cone)", l, re.I
    ))

    # Deduplicate (order-preserving)
    dedup = lambda xs: list(dict.fromkeys(xs))

    return {
        "steps": dedup(steps)[:12],
        "materials": dedup(mats)[:10],
        "tools": dedup(tools)[:10],
        "safety": dedup(safety)[:10]
    }
