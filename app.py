# app.py
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import re, math

from advice_engine.pipeline import run_advice_engine

app = FastAPI(title="Financial Advice Engine")
templates = Jinja2Templates(directory="templates")

# ---------- helpers ----------

def fmt_money(x: float) -> str:
    try:
        return f"₹{int(round(float(x))):,}"
    except Exception:
        return str(x)

def clean_result_for_view(d: dict) -> dict:
    """Pretty-print numbers, drop empty goal fields."""
    out = {}
    for k, v in d.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            v = None
        out[k] = v

    # hide goal fields if missing/blank
    if not out.get("Goal_Prob"):
        out.pop("Goal_Prob", None)
        out.pop("Goal_Label", None)

    # pretty print predicted savings
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
    """Extract income, spend, desired savings (monthly)."""
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
    if spend  is not None: spend  = _to_monthly(spend,  period_hint)
    if desired is not None: desired = _to_monthly(desired, period_hint)

    return {"income": income, "spend": spend, "desired": desired}

def make_advice(row: pd.Series) -> str:
    """Simple rule-based financial advice with formatting."""
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
            cut_amt = min(cut_amt, gap)  # cap at gap
            return f"Reduce {top_cat} by {fmt_money(cut_amt)} to close the ~{fmt_money(gap)} savings gap."
        else:
            return "You're on track for your savings goal. Maintain current habits and build a 1–2 month buffer."
    else:
        return f"Trim {top_cat} by 10% ({fmt_money(cut_amt)}) to boost monthly savings without major lifestyle changes."

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
    df = pd.DataFrame([row])
    out = run_advice_engine(df)
    cols = ["Pred_Savings_XGB"] + [c for c in ["Cluster", "Persona", "Goal_Prob", "Goal_Label"] if c in out.columns]
    result = clean_result_for_view(out[cols].iloc[0].to_dict())
    advice = make_advice(out.iloc[0])
    return templates.TemplateResponse("index.html", {"request": request, "result": result, "advice": advice})
