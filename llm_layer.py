# llm_layer.py
"""
LangChain-powered 'LLM layer' (LOCAL ONLY, no external APIs).
- Extracts structured fields from user text via a local HF model (flan-t5-base)
- Runs your ML pipeline (regression + clustering)
- Retrieves KB snippets from Chroma (rag.KBRetriever)
- Optionally crafts a short natural-language explanation via the same local model
- Handles general financial questions without requiring income/expense data

You can keep Streamlit exactly the same; just call AdvisorLLM().answer(query)
"""

from typing import Optional, Dict, Any, Union
import math
import pandas as pd
import logging
import re
import os
import torch  # <-- added

# Your existing modules
from advice_engine.pipeline import run_advice_engine
from rag import KBRetriever

# --- LangChain + local HuggingFace model ---
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline as hf_pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# 1) Output schema for parsing
# -------------------------
class ParsedQuery(BaseModel):
    income_value: Optional[float] = Field(None, description="Income numeric value")
    income_period: Optional[str]  = Field(None, description="income period: 'monthly' or 'annual'")
    spend_value: Optional[float]  = Field(None, description="Spending/expenses numeric value")
    spend_period: Optional[str]   = Field(None, description="spend period: 'monthly' or 'annual'")
    goal_value: Optional[float]   = Field(None, description="Desired savings numeric value if present")
    goal_period: Optional[str]    = Field(None, description="goal period: 'monthly' or 'annual' if present")

    @validator("income_period", "spend_period", "goal_period")
    def norm_period(cls, v):
        if v is None:
            return None
        v = str(v).lower().strip()
        if "year" in v or "annual" in v or v in {"y", "yr", "year", "annum"}:
            return "annual"
        if "month" in v or v in {"m", "mo", "mon", "monthly"}:
            return "monthly"
        return None

# -------------------------
# Fallback regex parser (from your original code)
# -------------------------
def parse_nl_query(text: str) -> dict:
    """
    Extract (monthly) income, spend, goal from a free-text question.
    Handles k/M suffixes and annual phrasing.
    """
    s = text.replace(",", "").replace("₹", "").strip().lower()
    period_hint = "year" if re.search(r"\b(per\s*year|annually|a\s*year|per\s*annum|/y|yr|year)\b", s) else None

    income, spend, desired = None, None, None
    # targeted patterns
    for m in re.finditer(r"\b(earn|income|salary|take\s*home)\s+([0-9\.]+[kKmM]?)", s):
        income = _token_to_num(m.group(2))
    for m in re.finditer(r"\b(spend|expense|expenses|costs|outgo)\s+([0-9\.]+[kKmM]?)", s):
        spend = _token_to_num(m.group(2))
    for m in re.finditer(r"\b(save|goal|target)\s+(?:to\s+)?(?:save\s+)?([0-9\.]+[kKmM]?)", s):
        desired = _token_to_num(m.group(2))

    # fallback: first two numbers are income/spend
    if income is None or spend is None:
        nums = re.findall(r"([0-9]*\.?[0-9]+[kKmM]?)", s)
        if len(nums) >= 2:
            income = income or _token_to_num(nums[0])
            spend  = spend  or _token_to_num(nums[1])
        if len(nums) >= 3 and desired is None:
            desired = _token_to_num(nums[2])

    if income is not None: income = _to_monthly(income, period_hint)
    if spend  is not None: spend  = _to_monthly(spend,  period_hint)
    if desired is not None: desired = _to_monthly(desired, period_hint)

    return {"income": income, "spend": spend, "desired": desired}

def _token_to_num(tok: str) -> float | None:
    m = re.match(r"([0-9]*\.?[0-9]+)\s*([kKmM])?$", tok.strip())
    if not m:
        return None
    val = float(m.group(1))
    suf = (m.group(2) or "").lower()
    if suf == "k": val *= 1_000
    if suf == "m": val *= 1_000_000
    return val

# -------------------------
# 2) Local LLM (flan-t5-base) — CPU-safe loading
# -------------------------
class _LocalLLM:
    _singleton = None

    @classmethod
    def get(cls):
        if cls._singleton is not None:
            return cls._singleton

        import os, torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline as hf_pipeline
        from langchain_huggingface import HuggingFacePipeline

        model_name = "google/flan-t5-small"  # light + free
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        auth = {"token": token} if token else {}

        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, **auth)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
            **auth,
        )
        mdl.to("cpu")

        gen = hf_pipeline(
            "text2text-generation",
            model=mdl,
            tokenizer=tok,
            device=-1,          # CPU
            max_new_tokens=256,
            max_length=512,
        )
        cls._singleton = HuggingFacePipeline(pipeline=gen)
        return cls._singleton

# -------------------------
# 3) Chains: parse + render + knowledge
# -------------------------
def _build_parse_chain():
    llm = _LocalLLM.get()
    parser = JsonOutputParser(pydantic_object=ParsedQuery)

    # Simplified prompt to reduce token length
    template = """Extract income, spend, and goal numbers from this question:

{question}

Format as JSON:
{format_instructions}

For example, "I earn 80k/year and spend 50k" would be:
{"income_value": 80000, "income_period": "annual", "spend_value": 50000}
"""
    prompt = PromptTemplate(
        template=template,
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt | llm | parser

def _build_render_chain():
    llm = _LocalLLM.get()
    template = """Write 2-3 sentences of financial advice based on:
- predicted_savings: {predicted_savings}
- persona: {persona}
- context: {kb_points}"""
    prompt = PromptTemplate.from_template(template)
    return prompt | llm

def _build_knowledge_chain():
    llm = _LocalLLM.get()
    template = """Answer this financial question using the knowledge provided:
Question: {question}

Knowledge context:
{kb_points}

Write a helpful 2-3 sentence answer:"""
    prompt = PromptTemplate.from_template(template)
    return prompt | llm

# -------------------------
# 4) Utility: unify to monthly
# -------------------------
def _to_monthly(value: Optional[float], period: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    if period == "annual" or period == "year":
        return float(value) / 12.0
    return float(value)

def _fmt_money(x: Any) -> str:
    try:
        return f"₹{int(round(float(x))):,}"
    except Exception:
        return str(x)

# -------------------------
# 5) Question type detection
# -------------------------
def _detect_question_type(query: str) -> str:
    """
    Detect the type of financial question being asked:
    - financial_analysis: Requires income/expense data (default)
    - knowledge_query: General financial knowledge
    - goal_planning: Savings goals
    - insurance_risk: Insurance/risk management
    - emergency_debt: Emergency funds or debt management
    """
    query_lower = query.lower()
    
    # Knowledge query patterns
    knowledge_patterns = [
        r"^(?:what|how|why|explain|describe|tell me about)\s",
        r"\bvs\b|\bversus\b|\bdifference\b",
        r"\brule\b|\bmethod\b|\bstrategy\b",
        r"\bbenefits?\b|\badvantages?\b|\bdisadvantages?\b"
    ]
    
    # Goal planning patterns
    goal_patterns = [
        r"\bgoal\b|\btarget\b|\bplan\b|\bplanning\b",
        r"\bsave (?:for|enough)\b",
        r"\bcan i reach\b|\bachieve\b",
        r"\bretirement\b|\blong[ -]term\b",
        r"\bage\b|\bfuture\b"
    ]
    
    # Insurance/risk patterns
    insurance_patterns = [
        r"\binsurance\b|\binsured\b",
        r"\brisk\b|\bprotection\b|\bcover\b",
        r"\bhealth\b|\blife\b|\bliability\b",
        r"\bdependents\b|\bfamily\b|\bchildren\b"
    ]
    
    # Emergency fund/debt patterns
    emergency_debt_patterns = [
        r"\bemergency\b|\brainy day\b",
        r"\bdebt\b|\bloan\b|\bcredit\b|\bemt\b",
        r"\bprioritize\b|\bfirst\b|\bsnowball\b|\bavalanche\b"
    ]

    # Financial profile patterns
    profile_patterns = [
        r"\bfinancial health\b|\bprofile\b|\bsummari[zs]e\b",
        r"\bpersona\b|\bcluster\b|\bprofile\b",
        r"\bmy financial\b|\bhealth check\b"
    ]
    
    # Check for basic financial analysis (must include income OR expense terms)
    has_income_terms = bool(re.search(r"\b(income|earn|salary|make|take home)\b", query_lower))
    has_expense_terms = bool(re.search(r"\b(spend|expense|cost|outgo)\b", query_lower))
    has_financial_data = bool(re.search(r"\d", query))
    
    # Check for specific question types
    for pattern in knowledge_patterns:
        if re.search(pattern, query_lower) and not (has_income_terms and has_expense_terms):
            return "knowledge_query"
    
    for pattern in goal_patterns:
        if re.search(pattern, query_lower):
            return "goal_planning"
            
    for pattern in insurance_patterns:
        if re.search(pattern, query_lower):
            return "insurance_risk"
            
    for pattern in emergency_debt_patterns:
        if re.search(pattern, query_lower):
            return "emergency_debt"

    for pattern in profile_patterns:
        if re.search(pattern, query_lower):
            return "financial_analysis"

    # Default to financial analysis
    return "financial_analysis"

# -------------------------
# 6) Orchestrator
# -------------------------
class AdvisorLLM:
    def __init__(self):
        self.parse_chain = _build_parse_chain()
        self.render_chain = _build_render_chain()
        self.knowledge_chain = _build_knowledge_chain()
        self.retriever = KBRetriever()

    def _row_from_parsed_llm(self, pq: ParsedQuery) -> Optional[Dict[str, Any]]:
        inc_mo = _to_monthly(pq.income_value, pq.income_period)
        spd_mo = _to_monthly(pq.spend_value, pq.spend_period)
        dsv_mo = _to_monthly(pq.goal_value, pq.goal_period)
        if inc_mo is None or spd_mo is None:
            return None
        return {
            "Income": inc_mo,
            "Groceries": spd_mo * 0.50,
            "Transport": spd_mo * 0.25,
            "Entertainment": spd_mo * 0.25,
            "Disposable_Income": max(0.0, inc_mo - spd_mo),
            "Desired_Savings": dsv_mo,
            "Occupation": None,
            "City_Tier": None,
        }
    
    def _row_from_parsed_regex(self, parsed: dict) -> Optional[Dict[str, Any]]:
        inc, spd, dsv = parsed.get("income"), parsed.get("spend"), parsed.get("desired")
        if inc is None or spd is None:
            return None
        disp = max(0.0, inc - spd)
        return {
            "Income": inc,
            "Groceries": spd * 0.50,
            "Transport": spd * 0.25,
            "Entertainment": spd * 0.25,
            "Disposable_Income": disp,
            "Desired_Savings": dsv,
            "Occupation": None,
            "City_Tier": None,
        }
    
    def _handle_knowledge_query(self, query: str, k_snippets: int = 3) -> Dict[str, Any]:
        """Handle general financial knowledge questions"""
        logger.info(f"Handling as knowledge query: {query}")
        
        # Retrieve relevant knowledge
        snippets = self.retriever.search(query, k=k_snippets)
        logger.info(f"Retrieved {len(snippets)} KB snippets for knowledge query")
        
        if not snippets:
            return {
                "error": "I couldn't find relevant information about this topic.",
                "result": None, "snippets": [], "advice": None
            }
        
        # Format knowledge points
        kb_points = "\n".join([f"- {s['text'][:300]}" for s in snippets[:3]])
        
        try:
            # Generate answer based on knowledge
            answer = self.knowledge_chain.invoke({
                "question": query,
                "kb_points": kb_points
            })
            
            # Create a simple result for consistency in UI
            result = {
                "Question_Type": "knowledge_query", 
                "Topic": snippets[0].get("metadata", {}).get("title", "Financial Knowledge")
            }
            
            return {
                "result": result,
                "snippets": snippets,
                "advice": str(answer)
            }
        except Exception as e:
            logger.error(f"Knowledge chain failed: {e}")
            # Return a fallback answer using the first snippet
            if snippets:
                return {
                    "result": {"Question_Type": "knowledge_query"},
                    "snippets": snippets,
                    "advice": snippets[0]["text"][:300]
                }
            else:
                return {
                    "error": "I couldn't process your knowledge question. Please try rephrasing.",
                    "result": None, "snippets": [], "advice": None
                }
    
    def _handle_goal_planning(self, query: str, parsed_data: dict, k_snippets: int = 3) -> Dict[str, Any]:
        """Handle goal planning questions with possibly partial financial data"""
        logger.info(f"Handling as goal planning with data: {parsed_data}")
        
        # See what data we have
        income = parsed_data.get("income")
        spend = parsed_data.get("spend")
        goal = parsed_data.get("desired")
        
        # If we have a goal but not income/spend, use knowledge-based approach
        if goal is not None and (income is None or spend is None):
            goal_text = f"₹{int(goal):,} per month"
            # Retrieve goal-specific knowledge
            snippets = self.retriever.search(f"savings goal {goal_text} financial planning", k=k_snippets)
            
            kb_points = "\n".join([f"- {s['text'][:250]}" for s in snippets[:3]])
            
            try:
                # Generate a response about the goal without income/spend data
                goal_prompt = f"""
                The user wants to save {goal_text}. They haven't provided current income and expenses.
                Based on this knowledge, what advice can you give them?
                
                Knowledge:
                {kb_points}
                
                Provide 2-3 sentences of practical advice:
                """
                
                llm = _LocalLLM.get()
                answer = llm.invoke(goal_prompt)
                
                return {
                    "result": {
                        "Question_Type": "goal_planning",
                        "Goal_Value": goal_text
                    },
                    "snippets": snippets,
                    "advice": str(answer)
                }
            except Exception as e:
                logger.error(f"Goal planning failed: {e}")
                # Fallback advice
                return {
                    "result": {"Question_Type": "goal_planning", "Goal_Value": goal_text},
                    "snippets": snippets,
                    "advice": f"To save {goal_text}, you'll need to ensure your income exceeds your expenses by at least that amount. Track your spending and find areas to reduce expenses."
                }
        
        # If we have no financial data at all, use pure knowledge approach
        if income is None and spend is None and goal is None:
            return self._handle_knowledge_query(query, k_snippets)
            
        # If we have income/spend, continue with normal analysis
        return None  # Let the main flow handle it

    def _handle_insurance_risk(self, query: str, k_snippets: int = 3) -> Dict[str, Any]:
        """Handle insurance and risk management questions"""
        logger.info(f"Handling as insurance/risk query: {query}")
        
        # Extract potential key terms
        insurance_terms = []
        for term in ["life", "health", "car", "auto", "vehicle", "home", "property", "liability", "disability"]:
            if term in query.lower():
                insurance_terms.append(term)
                
        # Enhance search query with specific terms
        search_query = query
        if insurance_terms:
            search_query = f"{query} {' '.join(insurance_terms)} insurance"
            
        # Retrieve relevant knowledge
        snippets = self.retriever.search(search_query, k=k_snippets)
        
        if not snippets:
            return {
                "error": "I couldn't find specific information about this insurance topic.",
                "result": None, "snippets": [], "advice": None
            }
            
        # Format knowledge points
        kb_points = "\n".join([f"- {s['text'][:250]}" for s in snippets[:3]])
        
        try:
            # Generate insurance-specific answer
            insurance_prompt = f"""
            Answer this insurance/risk management question:
            {query}
            
            Based on these knowledge points:
            {kb_points}
            
            Provide specific advice in 2-3 sentences:
            """
            
            llm = _LocalLLM.get()
            answer = llm.invoke(insurance_prompt)
            
            return {
                "result": {
                    "Question_Type": "insurance_risk",
                    "Insurance_Type": ', '.join(insurance_terms) if insurance_terms else "general"
                },
                "snippets": snippets,
                "advice": str(answer)
            }
        except Exception as e:
            logger.error(f"Insurance handling failed: {e}")
            # Fallback to first snippet
            return {
                "result": {"Question_Type": "insurance_risk"},
                "snippets": snippets,
                "advice": snippets[0]["text"][:300]
            }

    def _handle_emergency_debt(self, query: str, k_snippets: int = 3) -> Dict[str, Any]:
        """Handle emergency fund and debt management questions"""
        logger.info(f"Handling as emergency fund/debt query: {query}")
        
        # Determine specific subtopic
        is_emergency = "emergency" in query.lower() or "rainy day" in query.lower()
        is_debt = any(term in query.lower() for term in ["debt", "loan", "credit", "repayment"])
        
        # Enhance search query
        search_query = query
        if is_emergency:
            search_query = f"{query} emergency fund planning"
        elif is_debt:
            search_query = f"{query} debt management strategy"
            
        # Retrieve relevant knowledge
        snippets = self.retriever.search(search_query, k=k_snippets)
        
        if not snippets:
            return {
                "error": "I couldn't find specific information about this financial topic.",
                "result": None, "snippets": [], "advice": None
            }
            
        # Format knowledge points
        kb_points = "\n".join([f"- {s['text'][:250]}" for s in snippets[:3]])
        
        try:
            # Generate specific answer
            prompt = f"""
            Answer this question about {"emergency funds" if is_emergency else "debt management"}:
            {query}
            
            Based on these knowledge points:
            {kb_points}
            
            Provide specific advice in 2-3 sentences:
            """
            
            llm = _LocalLLM.get()
            answer = llm.invoke(prompt)
            
            subtopic = "emergency_fund" if is_emergency else "debt_management" if is_debt else "general"
            
            return {
                "result": {
                    "Question_Type": "emergency_debt",
                    "Subtopic": subtopic
                },
                "snippets": snippets,
                "advice": str(answer)
            }
        except Exception as e:
            logger.error(f"Emergency/debt handling failed: {e}")
            # Fallback to first snippet
            return {
                "result": {"Question_Type": "emergency_debt"},
                "snippets": snippets,
                "advice": snippets[0]["text"][:300]
            }

    def answer(self, user_query: str, k_snippets: int = 3) -> Dict[str, Any]:
        # First detect question type
        question_type = _detect_question_type(user_query)
        logger.info(f"Detected question type: {question_type}")
        
        # For non-financial analysis questions, use specialized handlers
        if question_type == "knowledge_query":
            return self._handle_knowledge_query(user_query, k_snippets)
            
        if question_type == "insurance_risk":
            return self._handle_insurance_risk(user_query, k_snippets)
            
        if question_type == "emergency_debt":
            return self._handle_emergency_debt(user_query, k_snippets)
        
        # For goal planning, extract any data first
        parsed_regex = parse_nl_query(user_query)
        if question_type == "goal_planning":
            result = self._handle_goal_planning(user_query, parsed_regex, k_snippets)
            if result is not None:
                return result
        
        # For financial analysis or goal planning with income/spend data, continue with existing flow
        
        # Try LLM parsing first (the original working flow)
        try:
            logger.info("Attempting to parse with LLM...")
            parsed = self.parse_chain.invoke({"question": user_query})
            logger.info(f"LLM parse result: {parsed}")
            row = self._row_from_parsed_llm(parsed)
            parse_method = "llm"
        except Exception as e:
            logger.warning(f"LLM parsing failed: {e}. Using regex results...")
            row = self._row_from_parsed_regex(parsed_regex)
            parse_method = "regex"
            
        # If we couldn't extract both income and spend, return appropriate error
        if row is None:
            if question_type == "financial_analysis":
                return {
                    "error": "I couldn't extract both income and spend. Try: 'My annual income is 9.6L, expenses 6L'.",
                    "result": None, "snippets": [], "advice": None
                }
            else:
                # For goal planning without income/spend, use knowledge approach
                return self._handle_knowledge_query(user_query, k_snippets)

        # Run your ML pipeline with the extracted data
        df = pd.DataFrame([row])
        out = run_advice_engine(df)
        logger.info(f"Model output: {out.iloc[0].to_dict()}")

        # Build minimal result dict for UI
        cols = ["Pred_Savings_XGB", "Cluster", "Persona"]
        if "Goal_Prob" in out.columns: cols += ["Goal_Prob", "Goal_Label"]
        result = out[cols].iloc[0].to_dict()
        if result.get("Pred_Savings_XGB") is not None:
            result["Pred_Savings_XGB"] = _fmt_money(result["Pred_Savings_XGB"])
            
        # Add question type and parse method to result
        result["Question_Type"] = question_type
        result["Parse_Method"] = parse_method
        
        # For goal planning, add goal achievement if a desired savings was specified
        if question_type == "goal_planning" and row.get("Desired_Savings") is not None:
            goal = row["Desired_Savings"]
            result["Goal_Value"] = _fmt_money(goal)
            
            # Calculate if they can achieve their goal
            predicted = result.get("Pred_Savings_XGB", "0")
            predicted_value = float(predicted.replace("₹", "").replace(",", ""))
            goal_achievement = (predicted_value / goal) * 100 if goal > 0 else 0
            result["Goal_Achievement"] = f"{goal_achievement:.1f}%"
            result["Goal_Achievable"] = "Yes" if predicted_value >= goal else "No"

        # Retrieve KB snippets (RAG)
        persona = result.get("Persona")
        q_hint = f"{user_query} persona {persona}" if persona else user_query
        snippets = self.retriever.search(q_hint, k=k_snippets)
        logger.info(f"Retrieved {len(snippets)} KB snippets")

        # Generate advice based on results
        try:
            logger.info("Generating advice...")
            
            # For goal planning with a goal, use specialized advice
            if question_type == "goal_planning" and "Goal_Value" in result:
                kb_points = "; ".join(s["text"][:100].replace("\n", " ") for s in snippets[:2])
                prompt = f"""
                The user wants to save {result['Goal_Value']} per month.
                Based on their financial details, they can save approximately {result['Pred_Savings_XGB']} per month.
                This means they {result['Goal_Achievable'].lower()} can reach their goal.
                
                Additional context: {kb_points}
                
                Write 2-3 sentences of specific advice about achieving their savings goal:
                """
                
                llm = _LocalLLM.get()
                advice = str(llm.invoke(prompt)).strip()
            else:
                # Use the standard render chain for other financial analysis
                kb_points = "; ".join(s["text"][:100].replace("\n", " ") for s in snippets[:2])
                render = self.render_chain.invoke({
                    "predicted_savings": result.get("Pred_Savings_XGB"),
                    "persona": result.get("Persona") or "Unknown",
                    "kb_points": kb_points[:200] or "No additional context.",
                })
                advice = str(render).strip()
                
            logger.info(f"Generated advice: {advice}")
        except Exception as e:
            logger.error(f"Failed to generate advice: {e}")
            # Create a simple rule-based advice as fallback
            if "Pred_Savings_XGB" in result:
                savings = result["Pred_Savings_XGB"]
                if question_type == "goal_planning" and "Goal_Value" in result:
                    goal = result["Goal_Value"]
                    if result["Goal_Achievable"] == "Yes":
                        advice = f"Based on your financial information, you could save {savings} per month, which exceeds your goal of {goal}. You're on track!"
                    else:
                        advice = f"Your predicted savings of {savings} falls short of your goal of {goal}. Consider reducing discretionary expenses to bridge this gap."
                else:
                    advice = f"Based on your financial information, you could save approximately {savings} per month."
            else:
                advice = "I recommend tracking your expenses carefully to find opportunities for saving."

        return {"result": result, "snippets": snippets, "advice": advice}
