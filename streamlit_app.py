# streamlit_app.py
import streamlit as st

from llm_layer import AdvisorLLM  # <- NEW: lightweight LLM-ish orchestrator

# --- Page config ---
st.set_page_config(page_title="Financial Advice Engine", layout="centered")

st.title("💸 Financial Advice Engine")
st.markdown("Ask me about your income, expenses, or financial goals:")

# Keep a single retriever/LLM instance across reruns
if "advisor_llm" not in st.session_state:
    st.session_state.advisor_llm = AdvisorLLM()
llm = st.session_state.advisor_llm

query = st.text_area(
    "Example: 'How much can I save if I earn ₹80K per year and spend ₹50K?'",
    height=120
)

if st.button("Get Advice") and query.strip():
    out = llm.answer(query, k_snippets=3)

    if out.get("error"):
        st.error(out["error"])
    else:
        res = out["result"]

        st.subheader("📊 Prediction Results")
        st.write(f"**Predicted Savings:** {res.get('Pred_Savings_XGB')}")
        st.write(f"**Cluster:** {res.get('Cluster')}")
        st.write(f"**Persona:** {res.get('Persona')}")

        if res.get("Goal_Prob") is not None:
            st.write(f"**Goal Probability:** {res.get('Goal_Prob')}")
            st.write(f"**Goal Label:** {res.get('Goal_Label')}")

        if out.get("advice"):
            st.success(f"**Advice:** {out['advice']}")

        snippets = out.get("snippets", [])
        if snippets:
            st.subheader("📚 Why this advice?")
            for s in snippets:
                st.markdown(
                    f"- {s['text'][:250]}{'…' if len(s['text'])>250 else ''}\n"
                    f"  <div style='font-size:0.9em; opacity:.7;'>"
                    f"  Source: {s['title']} ({s['source']}#{s['chunk']})"
                    f"  </div>",
                    unsafe_allow_html=True
                )
