# rag.py
import chromadb
from sentence_transformers import SentenceTransformer

KB_DB_DIR = "vectordb"
KB_COLLECTION = "finance_kb"
EMBED_MODEL = "all-MiniLM-L6-v2"

def _infer_tags_from_source(source: str) -> set[str]:
    if not source:
        return set()
    s = source.lower()
    tags = set()
    if "budget" in s: tags |= {"budget", "50/30/20"}
    if "emergency" in s: tags |= {"emergency", "fund"}
    if "debt" in s: tags |= {"debt", "avalanche", "snowball"}
    if "insurance" in s: tags |= {"insurance"}
    if "retirement" in s: tags |= {"retirement"}
    return tags


class KBRetriever:
    def __init__(self, db_dir: str = KB_DB_DIR, collection: str = KB_COLLECTION, model: str = EMBED_MODEL):
        self.client = chromadb.PersistentClient(path=db_dir)
        self.coll = self.client.get_collection(collection)
        self.embedder = SentenceTransformer(model)

    def search(self, query: str, k: int = 4):
        qvec = self.embedder.encode([query], normalize_embeddings=True).tolist()
        n = max(10, k * 4)

        # Try to get distances; fall back if not supported
        try:
            raw = self.coll.query(
                query_embeddings=qvec,
                n_results=n,
                include=["documents", "metadatas", "distances"],
            )
            docs = raw.get("documents", [[]])[0]
            metas = raw.get("metadatas", [[]])[0]
            dists = raw.get("distances", [[]])[0] or [None] * len(docs)
        except Exception:
            raw = self.coll.query(query_embeddings=qvec, n_results=n)
            docs = raw.get("documents", [[]])[0]
            metas = raw.get("metadatas", [[]])[0]
            dists = [None] * len(docs)

        seen = set()
        q_words = {w for w in query.lower().split() if len(w) > 2}
        # Intent hints: treat profile/persona like budgeting context (keep KB on)
        budget_context = bool(q_words & {"save","saving","spend","budget","expenses","cost","profile","persona","cluster"})

        candidates = []
        for doc, meta, dist in zip(docs, metas, dists):
            if not doc:
                continue
            src = (meta or {}).get("source")
            chk = (meta or {}).get("chunk")
            key = (src, chk)
            if key in seen:
                continue
            seen.add(key)

            # lexical overlap
            text_words = set(doc.lower().split())
            overlap = len(q_words & text_words)

            # tag prior (from filename)
            tags = _infer_tags_from_source(src or "")

            # scoring: base = overlap
            score = float(overlap)

            # tag boosts: prefer budget/emergency when in budget context
            if budget_context:
                if "budget" in tags:     score += 2.0
                if "50/30/20" in tags:   score += 1.5
                if "emergency" in tags:  score += 1.0

            # soft penalties: down-rank retirement/insurance unless asked
            asked_retire   = bool(q_words & {"retire","retirement","pension","pf","nps"})
            asked_insurance= bool(q_words & {"insurance","health","life","term","cover"})
            asked_debt = bool(q_words & {"debt","loan","credit","interest","emi"})
            if ("debt" in tags) and not asked_debt:
                score -= 1.5
            if ("retirement" in tags) and not asked_retire:
                score -= 2.0
            if ("insurance" in tags) and not asked_insurance:
                score -= 1.0

            # gentle tie-breaker by distance (if available)
            dd = dist if (dist is not None) else 0.0
            candidates.append({
                "text": doc, "source": src, "chunk": chk,
                "title": (meta or {}).get("title") or "(kb)",
                "score": score, "dist": dd,
            })

        if not candidates:
            # Fallback: first k raw
            hits = []
            for i in range(min(k, len(docs))):
                meta = metas[i] or {}
                hits.append({
                    "text": docs[i],
                    "source": meta.get("source"),
                    "chunk": meta.get("chunk"),
                    "title": meta.get("title") or "(kb)",
                })
            return hits

        # rank: score desc, then distance asc
        candidates.sort(key=lambda r: (-r["score"], r["dist"]))
        return candidates[:k]

