"""Interface Streamlit minimale pour interroger le coffre Obsidian."""

from __future__ import annotations

import streamlit as st

from chat_backend import ChatBackendError, answer_question


st.set_page_config(page_title="Obsidian RAG")
st.title("Obsidian RAG")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "exchanges" not in st.session_state:
    st.session_state.exchanges = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Posez votre question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    succeeded = False
    try:
        response = answer_question(prompt, st.session_state.exchanges[-6:])
        succeeded = True
    except ChatBackendError as exc:
        response = f"Erreur : {exc}"

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

    if succeeded:
        st.session_state.exchanges.append((prompt, response))
        st.session_state.exchanges = st.session_state.exchanges[-6:]
