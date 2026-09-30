"""Streamlit chat UI for FinMate. Run: streamlit run app.py"""
import uuid
import streamlit as st
from agent import chat, collection

st.set_page_config(page_title="FinMate", page_icon="📈")
st.title("📈 FinMate")
st.caption("Company information from verified summaries and live tools. Not investment advice.")

with st.sidebar:
    st.subheader("Companies covered")
    for c in sorted(m["company"] for m in collection.get()["metadatas"]):
        st.write("•", c)

if "tid" not in st.session_state:
    st.session_state.tid = str(uuid.uuid4())
    st.session_state.msgs = []

for role, text in st.session_state.msgs:
    st.chat_message(role).write(text)

if q := st.chat_input("Ask about a company, a share price, or a currency conversion"):
    st.chat_message("user").write(q)
    with st.spinner("Thinking..."):
        r = chat(q, st.session_state.tid)
    with st.chat_message("assistant"):
        st.write(r["answer"])
        if r["sources"]:
            st.caption(f"Sources: {', '.join(r['sources'])}  |  faithfulness {r['score']:.2f}")
    st.session_state.msgs += [("user", q), ("assistant", r["answer"])]
