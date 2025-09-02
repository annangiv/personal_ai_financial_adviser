# app.py
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import re, math

from advice_engine.pipeline import run_advice_engine

# ---------------- RAG (inline; no extra files) ----------------
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

    def search(self, query: str, k: int = 3):
        qvec = self.embedder.encode([query], normalize_embeddings=True).tolist()
        n = max(10, k * 4)
        try:
            raw = self.coll.query(query_embeddings=qvec, n_results=n, include=["documents","metadatas","distances"])
            docs = raw.get("documents", [[]])[0]
            metas = raw.get("metadatas", [[]])[0]
            dists = raw.get("distances", [[]])[0] or [None]*len(docs)
        except Exception:
            raw = self.coll.query(query_embeddings=qvec, n_results=n)
            docs = raw.get("documents", [[]])[0]
            metas = raw.get("metadatas", [[]])[0]
            dists = [None]*len(docs)

        seen, candidates = set(), []
        q_words = {w for w in query.lower().split() if len(w) > 2}
        budget_ctx = bool(q_words & {"save","saving","spend","budget","expenses","cost","profile","persona","cluster"})
        asked_ret = bool(q_words & {"retire","retirement","pension","pf","nps"})
        asked_ins = bool(q_words & {"insurance","health","life","term","cover"})
        asked_debt= bool(q_words & {"debt","loan","credit","interest","emi"})

        for doc, meta, dist in zip(docs, metas, dists):
            if not doc: 
                continue
            src = (meta or {}).get("source")
            chk = (meta or {}).get("chunk")
            key = (src, chk)
            if key in seen: 
                continue
            seen.add(key)

            text_words = set(doc.lower().split())
            overlap = len(q_words & text_words)
            tags = _infer_tags_from_source(src or "")
            score = float(overlap)
            if budget_ctx:
                if "budget" in tags:    score += 2.0
                if "50/30/20" in tags:  score += 1.5
                if "emergency" in tags: score += 1.0
            if ("retirement" in tags) and not asked_ret: score -= 2.0
            if ("insurance"  in tags) and not asked_ins: score -= 1.0
            if ("debt"       in tags) and not asked_debt: score -= 1.5

            candidates.append({
                "text": doc, "source": src, "chunk": chk,
                "title": (meta or {}).get("title") or "(kb)",
                "score": score, "dist": dist if dist is not None else 0.0,
            })

        if not candidates:
            hits = []
            for i in range(min(k, len(docs))):
                meta = metas[i] or {}
                hits.append({"text": docs[i], "source": meta.get("source"), "chunk": meta.get("chunk"), "title": meta.get("title") or "(kb)"})
            return hits

        candidates.sort(key=lambda r: (-r["score"], r["dist"]))
        return candidates[:k]

# ---------------- FastAPI + UI ----------------
app = FastAPI(title="Financial Advice Engine")
templates = Jinja2Templates(directory="templates")
kb = None  # global retriever

@app.on_event("startup")
def _init_kb():
    global kb
    try:
        kb = KBRetriever()
        print("[KB] retriever ready.")
    except Exception as e:
        kb = None
        print(f"[KB] disabled: {e}")

# ---------- helpers ----------
def fmt_money(x: float) -> str:
    try:
        return f"₹{int(round(float(x))):,}"
    except Exception:
        return str(x)

def clean_result_for_view(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            v = None
        out[k] = v
    if not out.get("Goal_Prob"):
        out.pop("Goal_Prob", None)
        out.pop("Goal_Label", None)
    if out.get("Pred_Savings_XGB") is not None:
        out["Pred_Savings_XGB"] = fmt_money(out["Pred_Savings_XGB"])
    return out

def _token_to_num(tok: str) -> float | None:
    m = re.match(r"([0-9]*\.?[0-9]+)\s*([kmKM])?$", tok.strip())
    if not m:
        return None
    val = float(m.group(1))
    suf = (m.group(2) or "").lower()
    if suf == "k": val *= 1_000
    if suf == "m": val *= 1_000_000
    return val

def _to_monthly(val: float, period_hint: str | None) -> float:
    if period_hint and ("year" in period_hint or "annual" in period_hint or "/y" in period_hint):
        return val / 12.0
    return val

def parse_nl_query(text: str) -> dict:
    """Extract income, spend, desired (monthly) and age (optional)."""
    s = text.replace(",", "").replace("₹", "").strip().lower()
    period_hint = "year" if re.search(r"\b(per\s*year|annually|a\s*year|per\s*annum|/y)\b", s) else None

    income, spend, desired = None, None, None
    for m in re.finditer(r"\b(earn|income|take\s*home)\s+([0-9\.]+[kmKM]?)", s):
        income = _token_to_num(m.group(2))
    for m in re.finditer(r"\b(spend|expense|expenses|costs|outgo)\s+([0-9\.]+[kmKM]?)", s):
        spend = _token_to_num(m.group(2))
    for m in re.finditer(r"\b(save|goal|target)\s+(?:to\s+)?(?:save\s+)?([0-9\.]+[kmKM]?)", s):
        desired = _token_to_num(m.group(2))

    if income is None or spend is None:
        nums = re.findall(r"([0-9]*\.?[0-9]+[kmKM]?)", s)
        if len(nums) >= 2:
            income = income or _token_to_num(nums[0])
            spend  = spend  or _token_to_num(nums[1])
        if len(nums) >= 3 and desired is None:
            desired = _token_to_num(nums[2])

    if income is not None: income = _to_monthly(income, period_hint)
    if spend  is not None:  spend  = _to_monthly(spend,  period_hint)
    if desired is not None: desired = _to_monthly(desired, period_hint)

    age = None
    m = re.search(r"\b(age|i[' ]?m|i am)\s*(\d{1,2})\b", s)
    if m:
        try:
            age = int(m.group(2))
        except Exception:
            age = None

    return {"income": income, "spend": spend, "desired": desired, "age": age}

def make_advice(row: pd.Series) -> str:
    pred = float(row.get("Pred_Savings_XGB", 0) or 0)
    desired = row.get("Desired_Savings", None)

    g = float(row.get("Groceries", 0) or 0)
    t = float(row.get("Transport", 0) or 0)
    e = float(row.get("Entertainment", 0) or 0)

    cat_vals = {"Groceries": g, "Transport": t, "Entertainment": e}
    top_cat = max(cat_vals, key=cat_vals.get)
    cut_amt = 0.10 * cat_vals[top_cat]

    if desired is not None and pd.notna(desired):
        desired = float(desired)
        gap = max(0.0, desired - pred)
        if gap > 0:
            cut_amt = min(cut_amt, gap)
            return f"You're short by ~{fmt_money(gap)}. Reduce {top_cat} by {fmt_money(cut_amt)} and move it to savings."
        else:
            return "You're on track for your goal. Maintain habits and build a 1–2 month buffer."
    else:
        return f"Trim {top_cat} by 10% ({fmt_money(cut_amt)}) to boost monthly savings without major lifestyle changes."

def retrieve_snippets(query: str, k: int = 3):
    if kb is None or not query or not query.strip():
        return []
    try:
        hits = kb.search(query, k=k)
        return hits
    except Exception:
        return []

def detect_intent(q: str) -> str:
    ql = q.lower()
    if any(w in ql for w in ["goal", "target", "milestone"]): return "goals"
    if any(w in ql for w in ["insurance", "life cover", "term plan"]): return "insurance"
    if any(w in ql for w in ["summary", "summarize", "overall health", "financial health"]): return "summary"
    return "savings"

def respond_goals(query: str, parsed: dict):
    age = parsed.get("age")
    if age:
        kb_q = f"financial goals at age {age} emergency fund retirement planning investing insurance"
    else:
        kb_q = "financial goals by age emergency fund retirement planning investing insurance"
    snippets = retrieve_snippets(kb_q, k=3)
    tip = []
    if age is not None:
        if age < 30:   tip.append("Prioritize a 3–6 month emergency fund and start investing early.")
        elif age < 40: tip.append("Increase retirement contributions and add term life if you have dependents.")
        else:          tip.append("Max retirement contributions, reduce high-interest debt, review insurance cover.")
    tip.append("See ‘Why this advice?’ for details.")
    advice = " ".join(tip)
    return None, advice, snippets

def respond_insurance(query: str, parsed: dict | None = None):
    kb_q = "do I need life insurance term insurance how much coverage rule of thumb income multiplier dependents liabilities"
    snippets = retrieve_snippets(kb_q, k=3)

    ann_income = None
    if parsed:
        inc = parsed.get("income")
        if inc is not None:
            try:
                ann_income = float(inc) * 12.0
            except Exception:
                ann_income = None

    coverage_line = ""
    if ann_income:
        low = ann_income * 10
        high = ann_income * 15
        prem_cap = ann_income * 0.02
        coverage_line = (
            f" Suggested coverage range: {fmt_money(low)}–{fmt_money(high)} "
            f"(~10–15× annual income). Aim for premiums ≤ {fmt_money(prem_cap)} per year."
        )

    advice = (
        "If you have dependents or liabilities, consider term life; "
        "a common rule is 10–15× annual income."
        + (coverage_line if coverage_line else " See details below.")
    )

    return None, advice, snippets


def respond_summary(df_row: dict):
    df = pd.DataFrame([df_row])
    out = run_advice_engine(df)
    row = out.iloc[0]

    pred = float(row.get("Pred_Savings_XGB", 0) or 0)
    persona = row.get("Persona") or ""
    cluster = row.get("Cluster")
    desired = row.get("Desired_Savings")

    lines = [f"Predicted monthly savings: {fmt_money(pred)}."]
    if persona:
        lines.append(f"Persona: {persona} (cluster {cluster}).")
    if desired is not None and pd.notna(desired):
        gap = max(0.0, float(desired) - pred)
        if gap > 0: lines.append(f"You're short of your goal by ~{fmt_money(gap)}.")
        else:       lines.append("You're on track for your stated savings goal.")

    kb_q = "budget 50/30/20 emergency fund savings rate how to improve savings"
    snippets = retrieve_snippets(kb_q, k=3)

    advice = " ".join(lines) + " See ‘Why this advice?’ for next steps."
    cols = ["Pred_Savings_XGB"] + [c for c in ["Cluster", "Persona", "Goal_Prob", "Goal_Label"] if c in out.columns]
    result = clean_result_for_view(out[cols].iloc[0].to_dict())
    return result, advice, snippets

# ---------- routes ----------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None, "advice": None})

@app.get("/structured", response_class=HTMLResponse)
def structured_get(request: Request):
    return templates.TemplateResponse("structured.html", {"request": request, "result": None, "advice": None})

@app.post("/advise-structured", response_class=HTMLResponse)
def advise_structured(request: Request,
    Income: float = Form(...),
    Disposable_Income: float = Form(...),
    Groceries: float = Form(...),
    Transport: float = Form(...),
    Entertainment: float = Form(...),
    Desired_Savings: float | None = Form(None),
    Occupation: str | None = Form(None),
    City_Tier: str | None = Form(None),
):
    row = {
        "Income": Income, "Disposable_Income": Disposable_Income,
        "Groceries": Groceries, "Transport": Transport, "Entertainment": Entertainment,
        "Desired_Savings": Desired_Savings, "Occupation": Occupation, "City_Tier": City_Tier,
    }
    df = pd.DataFrame([row])
    out = run_advice_engine(df)
    cols = ["Pred_Savings_XGB"] + [c for c in ["Cluster", "Persona", "Goal_Prob", "Goal_Label"] if c in out.columns]
    result = clean_result_for_view(out[cols].iloc[0].to_dict())
    advice = make_advice(out.iloc[0])
    return templates.TemplateResponse("structured.html", {"request": request, "result": result, "advice": advice})

@app.post("/advise", response_class=HTMLResponse)
def advise(request: Request,
    Income: float = Form(...),
    Disposable_Income: float = Form(...),
    Groceries: float = Form(...),
    Transport: float = Form(...),
    Entertainment: float = Form(...),
    Desired_Savings: float | None = Form(None),
    Occupation: str | None = Form(None),
    City_Tier: str | None = Form(None),
):
    row = {
        "Income": Income, "Disposable_Income": Disposable_Income,
        "Groceries": Groceries, "Transport": Transport, "Entertainment": Entertainment,
        "Desired_Savings": Desired_Savings, "Occupation": Occupation, "City_Tier": City_Tier,
    }
    df = pd.DataFrame([row])
    out = run_advice_engine(df)
    cols = ["Pred_Savings_XGB"] + [c for c in ["Cluster", "Persona", "Goal_Prob", "Goal_Label"] if c in out.columns]
    result = clean_result_for_view(out[cols].iloc[0].to_dict())
    advice = make_advice(out.iloc[0])
    return templates.TemplateResponse("index.html", {"request": request, "result": result, "advice": advice})

@app.post("/ask", response_class=HTMLResponse)
def ask(request: Request, query: str = Form(...)):
    parsed = parse_nl_query(query)
    intent = detect_intent(query)

    if intent == "goals":
        result, advice, snippets = respond_goals(query, parsed)
        return templates.TemplateResponse("index.html", {"request": request, "result": result, "advice": advice, "snippets": snippets})

    if intent == "insurance":
        result, advice, snippets = respond_insurance(query)
        return templates.TemplateResponse("index.html", {"request": request, "result": result, "advice": advice, "snippets": snippets})

    if intent == "summary":
        inc, spd, dsv = parsed.get("income"), parsed.get("spend"), parsed.get("desired")
        if inc is None or spd is None:
            return templates.TemplateResponse("index.html", {"request": request, "result": {"Pred_Savings_XGB": "—"}, "advice": "To summarize your financial health, include income and spend (e.g., “I earn 80k and spend 50k”)."})
        disp = max(0.0, inc - spd)
        row = {
            "Income": inc, "Disposable_Income": disp,
            "Groceries": spd * 0.5, "Transport": spd * 0.25, "Entertainment": spd * 0.25,
            "Desired_Savings": dsv, "Occupation": None, "City_Tier": None,
        }
        result, advice, snippets = respond_summary(row)
        return templates.TemplateResponse("index.html", {"request": request, "result": result, "advice": advice, "snippets": snippets})

    # default: savings (existing)
    inc, spd, dsv = parsed.get("income"), parsed.get("spend"), parsed.get("desired")
    if inc is None or spd is None:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": {"Pred_Savings_XGB": "—"},
            "advice": "I couldn’t parse income/spend. Try: “I earn 80K and spend 50K per year.”"
        })

    disp = max(0.0, inc - spd)
    row = {
        "Income": inc, "Disposable_Income": disp,
        "Groceries": spd * 0.5, "Transport": spd * 0.25, "Entertainment": spd * 0.25,
        "Desired_Savings": dsv, "Occupation": None, "City_Tier": None,
    }
    out = run_advice_engine(pd.DataFrame([row]))
    cols = ["Pred_Savings_XGB"] + [c for c in ["Cluster", "Persona", "Goal_Prob", "Goal_Label"] if c in out.columns]
    result = clean_result_for_view(out[cols].iloc[0].to_dict())
    advice = make_advice(out.iloc[0])

    # KB query bias for savings/profile-style questions
    kb_query = query
    ql = query.lower()
    if any(t in ql for t in ["save","saving","spend","budget","expenses","cost","profile","persona","cluster"]):
        kb_query = "budget 50/30/20 saving expenses groceries transport entertainment emergency fund"
    snippets = retrieve_snippets(kb_query, k=3)

    return templates.TemplateResponse("index.html", {"request": request, "result": result, "advice": advice, "snippets": snippets})
