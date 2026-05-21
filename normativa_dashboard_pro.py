import streamlit as st
import os
import json
import pickle
import base64
import re
from datetime import datetime

import numpy as np
import faiss
import pytesseract
import fitz  # PyMuPDF
import cv2

from PIL import Image
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# =====================================================
# CONFIGURAZIONE
# =====================================================

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_FOLDER = os.path.join(BASE_DIR, "pdf_normativa")

CHUNK_INDEX_FILE = os.path.join(BASE_DIR, "indice_chunk.faiss")
CHUNK_META_FILE = os.path.join(BASE_DIR, "metadata_chunk.pkl")

DOC_INDEX_FILE = os.path.join(BASE_DIR, "indice_documenti.faiss")
DOC_META_FILE = os.path.join(BASE_DIR, "metadata_documenti.pkl")


CHUNK_INDEX_FILE = "indice_chunk.faiss"
CHUNK_META_FILE = "metadata_chunk.pkl"

DOC_INDEX_FILE = "indice_documenti.faiss"
DOC_META_FILE = "metadata_documenti.pkl"

ARCHIVE_FILE = "archivio_risposte.json"

BACKGROUND_FILE = "sfondo_trenitalia.jpg"
LOGO_FILE = "regionale.png"

OPENAI_MODEL = "gpt-4.1-mini"
EMBED_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 180
TOP_K_SEARCH = 90
TOP_K_CONTEXT = 24

os.makedirs(PDF_FOLDER, exist_ok=True)


# =====================================================
# STREAMLIT
# =====================================================

st.set_page_config(
    layout="wide",
    page_title="AI NORMA",
    page_icon="📘"
)


# =====================================================
# STILE
# =====================================================

def set_background(image_file: str):
    if not os.path.exists(image_file):
        return

    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        .block-container {{
            background: rgba(255,255,255,0.94);
            backdrop-filter: blur(6px);
            border-radius: 16px;
            padding: 26px;
        }}

        section[data-testid="stSidebar"] {{
            background-color:#f1d400;
        }}

        section[data-testid="stSidebar"] * {{
            color:black !important;
        }}

        h1, h2, h3, h4 {{
            color:#e10600;
        }}

        .box-risposta {{
            background: rgba(255,255,255,0.96);
            border-radius:14px;
            padding:24px;
            margin-top:18px;
            margin-bottom:18px;
            box-shadow:0 4px 15px rgba(0,0,0,0.16);
            color:black;
            line-height:1.55;
        }}

        .box-risposta * {{
            color:black !important;
        }}

        .documento-box {{
            background:#fff8c4;
            border-left:6px solid #e10600;
            padding:12px;
            margin-bottom:10px;
            border-radius:8px;
        }}

        .domanda {{
            text-align:center;
            font-weight:bold;
            color:black;
            font-size:23px;
            margin-top:20px;
        }}

        .archivio {{
            text-align:center;
            font-weight:bold;
            font-size:24px;
            color:black;
        }}

        .sidebar-destra {{
            background-color:#f1d400;
            padding:16px;
            border-radius:10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


set_background(BACKGROUND_FILE)


# =====================================================
# CACHE MODELLI
# =====================================================

@st.cache_resource
def load_search_model():
    return SentenceTransformer(EMBED_MODEL)


search_model = load_search_model()


def get_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =====================================================
# UTILITY TESTO
# =====================================================

def pulisci_testo(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


def normalizza_query(text: str) -> str:
    if not text:
        return ""

    text = text.lower()
    text = text.replace("’", "'")
    text = re.sub(r"[^a-zàèéìòù0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def estrai_articolo(text: str) -> str:
    patterns = [
        r"(art\.?\s*\d+[\/\-]?\w*)",
        r"(articolo\s+\d+[\/\-]?\w*)",
        r"(comma\s+\d+)",
        r"(commi\s+\d+\s*(?:e|,|-)?\s*\d*)",
        r"(capo\s+[IVXLC]+)",
        r"(titolo\s+[IVXLC]+)",
        r"(paragrafo\s+\d+)",
        r"(punto\s+\d+[\/\-]?\w*)",
        r"(allegato\s+[A-Z0-9]+)"
    ]

    trovati = []

    for p in patterns:
        matches = re.findall(p, text, flags=re.IGNORECASE)

        for m in matches:
            trovati.append(m.strip())

    return " | ".join(sorted(list(set(trovati))))


def estrai_riferimento_normativo(text: str, file_name: str = "") -> str:
    """
    Estrae riferimenti normativi reali anche da file generici tipo normativa.pdf.
    Cerca DPR, D.P.R., DLgs, Decreto, Legge, CCNL, Accordo, Disposizione, Circolare, ecc.
    """

    testo = text.replace("\n", " ")
    testo = re.sub(r"\s+", " ", testo)

    patterns = [
        r"(D\.?\s*P\.?\s*R\.?\s*n\.?\s*\d+\s*(?:del|\/|-)?\s*\d{2,4})",
        r"(DPR\s*n\.?\s*\d+\s*(?:del|\/|-)?\s*\d{2,4})",
        r"(D\.?\s*Lgs\.?\s*n\.?\s*\d+\s*(?:del|\/|-)?\s*\d{2,4})",
        r"(Decreto\s+Legislativo\s+n\.?\s*\d+\s*(?:del|\/|-)?\s*\d{2,4})",
        r"(Legge\s+n\.?\s*\d+\s*(?:del|\/|-)?\s*\d{2,4})",
        r"(L\.?\s*n\.?\s*\d+\s*(?:del|\/|-)?\s*\d{2,4})",
        r"(Regolamento\s+[A-Z0-9\/\.\-\s]{3,40})",
        r"(CCNL\s+[A-Z0-9\/\.\-\s]{3,60})",
        r"(Accordo\s+[A-Z0-9\/\.\-\s]{3,60})",
        r"(Disposizione\s+[A-Z0-9\/\.\-\s]{3,60})",
        r"(Circolare\s+[A-Z0-9\/\.\-\s]{3,60})",
        r"(Contratto\s+Collettivo\s+[A-Z0-9\/\.\-\s]{3,80})",
        r"(Testo\s+Unico\s+[A-Z0-9\/\.\-\s]{3,80})"
    ]

    trovati = []

    for p in patterns:
        matches = re.findall(p, testo, flags=re.IGNORECASE)
        for m in matches:
            m = m.strip(" .,-;:")
            if len(m) > 4:
                trovati.append(m)

    # Se non trova nulla, prova dal nome file
    if not trovati and file_name:
        nome = file_name.replace(".pdf", "").replace("_", " ").replace("-", " ")
        nome = nome.strip()

        if nome.lower() not in ["normativa", "normativa completa", "documento"]:
            trovati.append(nome)

    if not trovati:
        return ""

    # pulizia duplicati preservando ordine
    puliti = []
    seen = set()

    for r in trovati:
        key = normalizza_query(r)
        if key not in seen:
            seen.add(key)
            puliti.append(r)

    return " | ".join(puliti[:3])


def crea_label_riferimento(source: str, norm_ref: str = "", article: str = "", page: int = None) -> str:
    parti = []

    if norm_ref:
        parti.append(norm_ref)
    else:
        parti.append(source)

    if article:
        parti.append(article)

    if page:
        parti.append(f"pag. {page}")

    return " - ".join(parti)


def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = pulisci_testo(text)

    if len(text) <= size:
        return [text] if text else []

    chunks = []
    start = 0

    while start < len(text):
        end = start + size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += size - overlap

    return chunks


def cosine_normalize(vectors):
    vectors = np.array(vectors).astype("float32")
    faiss.normalize_L2(vectors)
    return vectors


# =====================================================
# OCR POTENZIATO
# =====================================================

def prepara_immagine_ocr(img):
    img_np = np.array(img)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    gray = cv2.equalizeHist(gray)

    gray = cv2.medianBlur(gray, 3)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return Image.fromarray(thresh)


def estrai_testo_pagina(page, page_num, file_name):
    digital_text = page.get_text("text")
    digital_text = pulisci_testo(digital_text)

    if len(digital_text) >= 80:
        return digital_text, "testo digitale"

    try:
        zoom = 3
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        img_ocr = prepara_immagine_ocr(img)

        ocr_text = pytesseract.image_to_string(
            img_ocr,
            lang="ita",
            config="--oem 3 --psm 6"
        )

        ocr_text = pulisci_testo(ocr_text)

        if len(ocr_text) >= 20:
            return ocr_text, "OCR immagine potenziato"

    except Exception as e:
        return "", f"OCR fallito: {e}"

    return "", "nessun testo"


# =====================================================
# INDICIZZAZIONE
# =====================================================

def indicizza_documenti():
    chunk_texts = []
    chunk_metadata = []

    doc_texts = []
    doc_metadata = []

    report = []

    pdf_files = [
        f for f in os.listdir(PDF_FOLDER)
        if f.lower().endswith(".pdf")
    ]

    for file_name in pdf_files:
        path = os.path.join(PDF_FOLDER, file_name)

        try:
            pdf = fitz.open(path)
        except Exception as e:
            report.append({
                "documento": file_name,
                "riferimento_normativo": "",
                "stato": "ERRORE",
                "dettaglio": str(e),
                "pagine": 0,
                "pagine_con_testo": 0,
                "chunk": 0,
                "ocr": 0
            })
            continue

        total_pages = len(pdf)
        full_doc_text = []
        doc_chunk_count = 0
        ocr_pages = 0
        pages_with_text = 0
        current_norm_ref = ""

        for page_num in range(total_pages):
            page = pdf[page_num]

            text, extraction_type = estrai_testo_pagina(
                page,
                page_num + 1,
                file_name
            )

            if "OCR" in extraction_type:
                ocr_pages += 1

            if not text or len(text.strip()) < 20:
                continue

            pages_with_text += 1
            full_doc_text.append(f"\n\n[PAGINA {page_num + 1}]\n{text}")

            page_norm_ref = estrai_riferimento_normativo(text[:2500], file_name)

            if page_norm_ref:
                current_norm_ref = page_norm_ref

            chunks = chunk_text(text)

            for chunk in chunks:
                article = estrai_articolo(chunk)

                chunk_norm_ref = estrai_riferimento_normativo(chunk, file_name)

                if chunk_norm_ref:
                    current_norm_ref = chunk_norm_ref

                effective_norm_ref = chunk_norm_ref or current_norm_ref or estrai_riferimento_normativo(file_name, file_name)

                reference_label = crea_label_riferimento(
                    source=file_name,
                    norm_ref=effective_norm_ref,
                    article=article,
                    page=page_num + 1
                )

                enriched_text = f"""
RIFERIMENTO NORMATIVO: {effective_norm_ref}
DOCUMENTO FILE: {file_name}
PAGINA: {page_num + 1}
ARTICOLO/RIFERIMENTO: {article}
ETICHETTA RIFERIMENTO: {reference_label}
TIPO ESTRAZIONE: {extraction_type}

TESTO:
{chunk}
"""

                chunk_texts.append(enriched_text)

                chunk_metadata.append({
                    "source": file_name,
                    "norm_ref": effective_norm_ref,
                    "reference_label": reference_label,
                    "page": page_num + 1,
                    "article": article,
                    "text": chunk,
                    "extraction": extraction_type
                })

                doc_chunk_count += 1

        pdf.close()

        complete_text = "\n".join(full_doc_text)
        complete_text = pulisci_testo(complete_text)

        doc_norm_ref = estrai_riferimento_normativo(complete_text[:8000], file_name)

        if complete_text:
            doc_texts.append(f"RIFERIMENTO NORMATIVO: {doc_norm_ref}\nDOCUMENTO FILE: {file_name}\n\n{complete_text[:15000]}")
            doc_metadata.append({
                "source": file_name,
                "norm_ref": doc_norm_ref,
                "pages": total_pages,
                "text_preview": complete_text[:2500],
                "chunk_count": doc_chunk_count,
                "ocr_pages": ocr_pages
            })

        report.append({
            "documento": file_name,
            "riferimento_normativo": doc_norm_ref,
            "stato": "OK" if doc_chunk_count > 0 else "VUOTO",
            "dettaglio": "Indicizzato" if doc_chunk_count > 0 else "Nessun testo estratto",
            "pagine": total_pages,
            "pagine_con_testo": pages_with_text,
            "chunk": doc_chunk_count,
            "ocr": ocr_pages
        })

    if not chunk_texts:
        return False, report

    chunk_embeddings = search_model.encode(
        chunk_texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    chunk_embeddings = cosine_normalize(chunk_embeddings)

    chunk_index = faiss.IndexFlatIP(chunk_embeddings.shape[1])
    chunk_index.add(chunk_embeddings)

    faiss.write_index(chunk_index, CHUNK_INDEX_FILE)

    with open(CHUNK_META_FILE, "wb") as f:
        pickle.dump(chunk_metadata, f)

    if doc_texts:
        doc_embeddings = search_model.encode(
            doc_texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        doc_embeddings = cosine_normalize(doc_embeddings)

        doc_index = faiss.IndexFlatIP(doc_embeddings.shape[1])
        doc_index.add(doc_embeddings)

        faiss.write_index(doc_index, DOC_INDEX_FILE)

        with open(DOC_META_FILE, "wb") as f:
            pickle.dump(doc_metadata, f)

    return True, report


# =====================================================
# CARICAMENTO INDICI
# =====================================================

def carica_indici():
    chunk_index = None
    chunk_metadata = []

    doc_index = None
    doc_metadata = []

    if os.path.exists(CHUNK_INDEX_FILE) and os.path.exists(CHUNK_META_FILE):
        chunk_index = faiss.read_index(CHUNK_INDEX_FILE)

        with open(CHUNK_META_FILE, "rb") as f:
            chunk_metadata = pickle.load(f)

    if os.path.exists(DOC_INDEX_FILE) and os.path.exists(DOC_META_FILE):
        doc_index = faiss.read_index(DOC_INDEX_FILE)

        with open(DOC_META_FILE, "rb") as f:
            doc_metadata = pickle.load(f)

    return chunk_index, chunk_metadata, doc_index, doc_metadata


chunk_index, chunk_metadata, doc_index, doc_metadata = carica_indici()


# =====================================================
# RICERCA IBRIDA POTENZIATA
# =====================================================

def keyword_score(query, text, source="", norm_ref="", article=""):
    q = normalizza_query(query)
    t = normalizza_query(text)
    s = normalizza_query(source)
    n = normalizza_query(norm_ref)
    a = normalizza_query(article)

    words = [w for w in q.split() if len(w) > 2]

    if not words:
        return 0

    score = 0

    for w in words:
        if w in t:
            score += 2
        if w in s:
            score += 3
        if w in n:
            score += 8
        if w in a:
            score += 8

    if q in t:
        score += 10

    if q in n:
        score += 25

    if q in a:
        score += 20

    return score


def costruisci_bm25(selected_docs=None):
    corpus = []
    valid_chunks = []

    for m in chunk_metadata:
        if selected_docs and m["source"] not in selected_docs:
            continue

        testo = f"""
        {m.get('norm_ref', '')}
        {m.get('reference_label', '')}
        {m['source']}
        {m.get('article', '')}
        {m['text']}
        """

        tokens = normalizza_query(testo).split()

        if tokens:
            corpus.append(tokens)
            valid_chunks.append(m)

    if not corpus:
        return None, [], []

    bm25 = BM25Okapi(corpus)

    return bm25, corpus, valid_chunks


def retrieve_relevant_context(query, selected_docs=None):
    if chunk_index is None or not chunk_metadata:
        return [], [], []

    query_emb = search_model.encode([query], convert_to_numpy=True)
    query_emb = cosine_normalize(query_emb)

    distances, ids = chunk_index.search(query_emb, TOP_K_SEARCH)

    bm25, corpus, valid_chunks = costruisci_bm25(selected_docs)
    bm25_scores_by_key = {}

    if bm25 is not None:
        query_tokens = normalizza_query(query).split()
        scores = bm25.get_scores(query_tokens)

        for i, score in enumerate(scores):
            chunk_key = (
                valid_chunks[i]["source"],
                valid_chunks[i].get("norm_ref", ""),
                valid_chunks[i]["page"],
                valid_chunks[i]["text"][:100]
            )
            bm25_scores_by_key[chunk_key] = float(score)

    candidates = []

    for rank, idx in enumerate(ids[0]):
        if idx < 0 or idx >= len(chunk_metadata):
            continue

        chunk = chunk_metadata[idx]

        if selected_docs and chunk["source"] not in selected_docs:
            continue

        semantic_score = float(distances[0][rank])

        chunk_key = (
            chunk["source"],
            chunk.get("norm_ref", ""),
            chunk["page"],
            chunk["text"][:100]
        )

        bm25_score = bm25_scores_by_key.get(chunk_key, 0)

        key_score = keyword_score(
            query,
            chunk["text"],
            chunk["source"],
            chunk.get("norm_ref", ""),
            chunk.get("article", "")
        )

        query_lower = query.lower()
        source_lower = chunk["source"].lower()
        norm_lower = chunk.get("norm_ref", "").lower()
        article_lower = chunk.get("article", "").lower()

        article_boost = 0
        if chunk.get("article"):
            article_boost += 6
            if any(w in query_lower for w in ["art", "articolo", "comma", "punto", "allegato"]):
                article_boost += 12

        norm_ref_boost = 0
        if chunk.get("norm_ref"):
            norm_ref_boost += 8

        for word in query_lower.split():
            word = word.strip()
            if len(word) > 3 and word in norm_lower:
                norm_ref_boost += 10
            if len(word) > 3 and word in article_lower:
                norm_ref_boost += 10

        title_boost = 0
        for word in query_lower.split():
            word = word.strip()
            if len(word) > 3 and word in source_lower:
                title_boost += 4

        ocr_boost = 0
        if "OCR" in chunk.get("extraction", ""):
            ocr_boost += 2

        exact_phrase_boost = 0
        if normalizza_query(query) in normalizza_query(chunk["text"]):
            exact_phrase_boost += 18

        final_score = (
            semantic_score * 100
            + bm25_score * 3.5
            + key_score
            + article_boost
            + norm_ref_boost
            + title_boost
            + ocr_boost
            + exact_phrase_boost
            - rank * 0.05
        )

        candidates.append({
            "chunk": chunk,
            "semantic_score": semantic_score,
            "bm25_score": bm25_score,
            "keyword_score": key_score,
            "final_score": final_score
        })

    if bm25 is not None:
        query_tokens = normalizza_query(query).split()
        scores = bm25.get_scores(query_tokens)

        top_bm25_idx = np.argsort(scores)[::-1][:50]

        for i in top_bm25_idx:
            chunk = valid_chunks[i]

            bm25_score = float(scores[i])

            key_score = keyword_score(
                query,
                chunk["text"],
                chunk["source"],
                chunk.get("norm_ref", ""),
                chunk.get("article", "")
            )

            query_lower = query.lower()
            source_lower = chunk["source"].lower()
            norm_lower = chunk.get("norm_ref", "").lower()
            article_lower = chunk.get("article", "").lower()

            title_boost = 0
            for word in query_lower.split():
                word = word.strip()
                if len(word) > 3 and word in source_lower:
                    title_boost += 4

            article_boost = 0
            if chunk.get("article"):
                article_boost += 6
                if any(w in query_lower for w in ["art", "articolo", "comma", "punto", "allegato"]):
                    article_boost += 12

            norm_ref_boost = 0
            if chunk.get("norm_ref"):
                norm_ref_boost += 8

            for word in query_lower.split():
                word = word.strip()
                if len(word) > 3 and word in norm_lower:
                    norm_ref_boost += 10
                if len(word) > 3 and word in article_lower:
                    norm_ref_boost += 10

            ocr_boost = 0
            if "OCR" in chunk.get("extraction", ""):
                ocr_boost += 2

            exact_phrase_boost = 0
            if normalizza_query(query) in normalizza_query(chunk["text"]):
                exact_phrase_boost += 18

            final_score = (
                bm25_score * 4.5
                + key_score
                + title_boost
                + article_boost
                + norm_ref_boost
                + ocr_boost
                + exact_phrase_boost
            )

            candidates.append({
                "chunk": chunk,
                "semantic_score": 0,
                "bm25_score": bm25_score,
                "keyword_score": key_score,
                "final_score": final_score
            })

    unique = {}

    for c in candidates:
        ch = c["chunk"]
        key = (
            ch["source"],
            ch.get("norm_ref", ""),
            ch["page"],
            ch["text"][:140]
        )

        if key not in unique or c["final_score"] > unique[key]["final_score"]:
            unique[key] = c

    candidates = list(unique.values())

    candidates = sorted(
        candidates,
        key=lambda x: x["final_score"],
        reverse=True
    )

    doc_scores = {}

    for c in candidates[:70]:
        chunk = c["chunk"]
        ref_key = chunk.get("reference_label") or crea_label_riferimento(
            chunk["source"],
            chunk.get("norm_ref", ""),
            chunk.get("article", ""),
            chunk.get("page")
        )

        if ref_key not in doc_scores:
            doc_scores[ref_key] = {
                "score": 0,
                "pages": set(),
                "ocr": 0,
                "chunks": 0,
                "source": chunk["source"],
                "norm_ref": chunk.get("norm_ref", ""),
                "article": chunk.get("article", "")
            }

        doc_scores[ref_key]["score"] += c["final_score"]
        doc_scores[ref_key]["pages"].add(chunk["page"])
        doc_scores[ref_key]["chunks"] += 1

        if "OCR" in chunk.get("extraction", ""):
            doc_scores[ref_key]["ocr"] += 1

    ranked_docs = sorted(
        doc_scores.items(),
        key=lambda x: x[1]["score"],
        reverse=True
    )

    top_refs = [d[0] for d in ranked_docs[:4]]

    final_results = []

    for c in candidates:
        chunk = c["chunk"]
        ref_key = chunk.get("reference_label") or crea_label_riferimento(
            chunk["source"],
            chunk.get("norm_ref", ""),
            chunk.get("article", ""),
            chunk.get("page")
        )

        if ref_key in top_refs:
            final_results.append(chunk)

    final_results = final_results[:TOP_K_CONTEXT]

    return final_results, top_refs, ranked_docs[:6]


# =====================================================
# RISPOSTA AI
# =====================================================

def generate_answer(query, results, ranked_docs):
    client = get_client()

    context = "\n\n---\n\n".join([
        f"""
RIFERIMENTO NORMATIVO: {r.get('norm_ref', '')}
ARTICOLO/RIFERIMENTO: {r.get('article', '')}
ETICHETTA RIFERIMENTO: {r.get('reference_label', '')}
DOCUMENTO FILE: {r['source']}
PAGINA: {r['page']}
TIPO ESTRAZIONE: {r.get('extraction', '')}

TESTO:
{r['text']}
"""
        for r in results
    ])

    docs_summary = "\n".join([
        f"- {ref}: punteggio {info['score']:.2f}, file {info.get('source','')}, pagine trovate {sorted(list(info['pages']))}"
        for ref, info in ranked_docs
    ])

    prompt = f"""
Sei un assistente esperto di normativa ferroviaria Trenitalia.

Devi rispondere SOLO usando i testi forniti.
Se il testo non basta, devi dirlo chiaramente.
Non inventare articoli, commi, pagine o documenti non presenti negli estratti.
Se il testo è OCR e sembra incompleto, segnalalo.

REGOLA IMPORTANTE:
Se il file si chiama normativa.pdf o ha un nome generico, NON usare "normativa.pdf" come riferimento principale.
Usa invece il campo RIFERIMENTO NORMATIVO, ARTICOLO/RIFERIMENTO o ETICHETTA RIFERIMENTO.
Il nome file va indicato solo come supporto tecnico.

DOMANDA UTENTE:
{query}

RIFERIMENTI PIÙ PERTINENTI TROVATI:
{docs_summary}

ESTRATTI NORMATIVI:
{context}

Produci una risposta in italiano, molto chiara e operativa, con questa struttura:

1. RIFERIMENTO NORMATIVO PRINCIPALE INDIVIDUATO
- indica DPR/Legge/CCNL/Accordo/Circolare se presente
- indica articolo, comma, punto se presente
- indica pagina
- indica tra parentesi il file tecnico solo se utile

2. COSA DICE LA NORMA
- spiega il contenuto normativo rilevante
- cita riferimento normativo e pagina durante la spiegazione

3. RISPOSTA OPERATIVA
- rispondi concretamente alla domanda dell'utente
- usa un linguaggio pratico da sala operativa / produzione
- specifica cosa fare e cosa verificare

4. RIFERIMENTI TROVATI
- riferimento normativo reale
- articolo/comma/punto se presente
- pagina
- file tecnico
- tipo estrazione: testo digitale o OCR

5. LIMITI / ATTENZIONE
- segnala se gli estratti sono parziali
- segnala se serve consultazione manuale del documento completo
"""

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt
    )

    return response.output_text


# =====================================================
# ARCHIVIO
# =====================================================

def salva_risposta(domanda, risposta, docs):
    voce = {
        "domanda": domanda,
        "risposta": risposta,
        "data": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "argomento": docs[0] if docs else "Altro"
    }

    if os.path.exists(ARCHIVE_FILE):
        with open(ARCHIVE_FILE, encoding="utf-8") as f:
            archivio = json.load(f)
    else:
        archivio = []

    archivio.append(voce)

    with open(ARCHIVE_FILE, "w", encoding="utf-8") as f:
        json.dump(archivio, f, ensure_ascii=False, indent=2)


# =====================================================
# SIDEBAR
# =====================================================

if os.path.exists(LOGO_FILE):
    st.sidebar.image(LOGO_FILE, use_container_width=True)

st.sidebar.markdown("## Carica documenti")

uploaded_files = st.sidebar.file_uploader(
    "Carica PDF normativi",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        path = os.path.join(PDF_FOLDER, file.name)

        with open(path, "wb") as f:
            f.write(file.getbuffer())

        st.sidebar.success(f"Caricato: {file.name}")

st.sidebar.markdown("---")

mostra_file_tecnico = st.sidebar.checkbox(
    "Mostra anche nome file tecnico",
    value=True
)

mostra_estratti = st.sidebar.checkbox(
    "Mostra estratti trovati",
    value=False
)

if st.sidebar.button("🔄 Indicizza documenti", type="primary"):
    with st.spinner("Indicizzazione in corso. Analizzo anche PDF scansionati con OCR potenziato e riferimenti normativi..."):
        ok, report = indicizza_documenti()

    if ok:
        st.success("Indicizzazione completata.")
        st.info("Ricarica la pagina se non vedi subito i nuovi documenti nella lista.")
    else:
        st.error("Nessun contenuto indicizzabile trovato.")

    st.markdown("### Report indicizzazione")
    st.dataframe(report, use_container_width=True)

    st.stop()


chunk_index, chunk_metadata, doc_index, doc_metadata = carica_indici()

all_docs = sorted(list({m["source"] for m in chunk_metadata}))

st.sidebar.markdown("## Documenti indicizzati")

st.sidebar.caption(f"Documenti trovati: {len(all_docs)}")
st.sidebar.caption(f"Chunk indicizzati: {len(chunk_metadata)}")

selected_docs = st.sidebar.multiselect(
    "Limita ricerca a questi documenti",
    all_docs
)

with st.sidebar.expander("📄 Elenco documenti indicizzati"):
    if all_docs:
        for d in all_docs:
            doc_chunks = [m for m in chunk_metadata if m["source"] == d]
            count = len(doc_chunks)
            ocr_count = len([
                m for m in doc_chunks
                if "OCR" in m.get("extraction", "")
            ])
            refs = sorted(list({
                m.get("norm_ref", "")
                for m in doc_chunks
                if m.get("norm_ref")
            }))

            st.markdown(f"**{d}**")
            st.caption(f"Chunk: {count} | OCR: {ocr_count}")

            if refs:
                st.caption("Riferimenti riconosciuti:")
                for r in refs[:5]:
                    st.caption(f"- {r}")
    else:
        st.warning("Nessun documento indicizzato.")


st.sidebar.markdown("---")
st.sidebar.markdown("## Riassumi documento")

doc_to_summarize = st.sidebar.selectbox(
    "Scegli documento",
    [""] + all_docs
)

if "summary_output" not in st.session_state:
    st.session_state.summary_output = ""

if st.sidebar.button("Riassumi documento") and doc_to_summarize:
    with st.spinner("Sto riassumendo il documento..."):
        client = get_client()

        doc_chunks = [
            m for m in chunk_metadata
            if m["source"] == doc_to_summarize
        ]

        text = "\n\n".join([
            f"Riferimento: {c.get('reference_label','')} - Pagina {c['page']} - {c['text']}"
            for c in doc_chunks[:120]
        ])

        prompt = f"""
Riassumi il seguente documento normativo ferroviario.

DOCUMENTO TECNICO:
{doc_to_summarize}

TESTO:
{text}

Struttura:
1. OGGETTO
2. CONTENUTO PRINCIPALE
3. RIFERIMENTI NORMATIVI INDIVIDUATI
4. ARTICOLI/PAGINE IMPORTANTI
5. POSSIBILI DOMANDE A CUI IL DOCUMENTO RISPONDE
"""

        response = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt
        )

        st.session_state.summary_output = response.output_text


# =====================================================
# LAYOUT
# =====================================================

col_centro, col_destra = st.columns([4, 1.25])


with col_centro:

    if os.path.exists(BACKGROUND_FILE):
        st.image(BACKGROUND_FILE, use_container_width=True)

    st.markdown(
        """
        <h1>AI NORMA</h1>
        <h3>Direzione Operations Regionale</h3>
        <h4>Direzione Regionale Sicilia</h4>
        <p><b>Realizzato da:</b> Marco Failla - Simone Rinaldi</p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    if st.session_state.summary_output:
        st.markdown(
            f"""
            <div class="box-risposta">
            {st.session_state.summary_output}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<div class="domanda">DOMANDA</div>', unsafe_allow_html=True)

    query = st.text_input(
        "",
        placeholder="Scrivi una domanda normativa...",
        label_visibility="collapsed"
    )

    if query:
        with st.spinner("Ricerca normativa ibrida potenziata per articoli e riferimenti..."):
            results, docs, ranked_docs = retrieve_relevant_context(
                query,
                selected_docs=selected_docs
            )

            if results:
                answer = generate_answer(query, results, ranked_docs)
                salva_risposta(query, answer, docs)
            else:
                answer = """
Nessun risultato pertinente trovato.

Possibili cause:
- i documenti non sono stati ancora indicizzati;
- il PDF è una scansione e l'OCR non è riuscito;
- la domanda usa termini molto diversi da quelli presenti nel documento.
"""

        st.markdown("### Riferimenti più pertinenti trovati")

        if ranked_docs:
            for ref, info in ranked_docs:
                pages = sorted(list(info["pages"]))

                file_info = ""
                if mostra_file_tecnico:
                    file_info = f"<br>File tecnico: {info.get('source', '')}"

                st.markdown(
                    f"""
                    <div class="documento-box">
                    <b>{ref}</b><br>
                    Punteggio ricerca: {info['score']:.2f}<br>
                    Pagine rilevanti: {pages}<br>
                    Chunk OCR: {info['ocr']}<br>
                    Chunk usati nel ranking: {info['chunks']}
                    {file_info}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        if mostra_estratti and results:
            with st.expander("🔎 Estratti normativi usati per la risposta"):
                for r in results[:12]:
                    st.markdown(f"**{r.get('reference_label', '')}**")
                    st.caption(f"File: {r['source']} | Pagina: {r['page']} | Estrazione: {r.get('extraction', '')}")
                    st.write(r["text"][:1200])
                    st.markdown("---")

        st.markdown(
            f"""
            <div class="box-risposta">
            {answer}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.download_button(
            label="🖨️ Scarica risposta",
            data=answer,
            file_name="risposta_normativa.txt",
            mime="text/plain"
        )


with col_destra:

    st.markdown('<div class="sidebar-destra">', unsafe_allow_html=True)
    st.markdown('<div class="archivio">ARCHIVIO RISPOSTE</div>', unsafe_allow_html=True)

    if os.path.exists(ARCHIVE_FILE):
        with open(ARCHIVE_FILE, encoding="utf-8") as f:
            data = json.load(f)

        archivio = {}

        for item in data:
            arg = item.get("argomento", "Altro")
            archivio.setdefault(arg, [])
            archivio[arg].append(item)

        for arg, items in archivio.items():
            with st.expander(arg):
                for r in reversed(items):
                    st.markdown(f"**{r['domanda']}**")
                    st.caption(r["data"])

                    st.markdown(
                        f"""
                        <div class="box-risposta">
                        {r["risposta"]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown("---")
    else:
        st.caption("Ancora nessuna risposta salvata.")

    st.markdown('</div>', unsafe_allow_html=True)



