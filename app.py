import os
import uuid
import streamlit as st
from langchain_core.messages import ToolMessage
from agent import build_graph

st.set_page_config(page_title="Study Day Assistant", page_icon="📚")
st.title("📚 Study Day Assistant")
st.caption("Manage your study tasks, check Tbilisi weather, and find study books with safe approval controls.")

# API Key-ის წაკითხვა Secrets-იდან ან გარემოს ცვლადიდან
if "GROQ_API_KEY" not in os.environ:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    else:
        api_key = st.sidebar.text_input("Enter GROQ API Key", type="password")
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key

if not os.environ.get("GROQ_API_KEY"):
    st.warning("Please provide your GROQ_API_KEY in secrets or the sidebar to start.")
    st.stop()

# სესიის ინიციალიზაცია
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_approval" not in st.session_state:
    st.session_state.pending_approval = None

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# საუბრის ისტორიის ჩვენება
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# თუ აგენტი ელოდება დადასტურებას (Approval Flow)
if st.session_state.pending_approval:
    info = st.session_state.pending_approval
    st.warning(f"⚠️ Are you sure you want to delete task: **{info['task']}**?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve", use_container_width=True):
            try:
                resumed = st.session_state.graph.invoke(None, config)
                reply = resumed["messages"][-1].content
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"API Error: {e}")
            st.session_state.pending_approval = None
            st.rerun()
    with col2:
        if st.button("❌ Deny", use_container_width=True):
            try:
                cancel_msg = ToolMessage(
                    tool_call_id=info["tool_call_id"],
                    content="Action cancelled by user. The task was NOT deleted."
                )
                st.session_state.graph.update_state(
                    config, {"messages": [cancel_msg]}, as_node="sensitive_tools"
                )
                resumed = st.session_state.graph.invoke(None, config)
                reply = resumed["messages"][-1].content
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"API Error: {e}")
            st.session_state.pending_approval = None
            st.rerun()

# მომხმარებლის ახალი შეტყობინება
if prompt := st.chat_input("Ask me to add a task, check Tbilisi weather, or find books..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        result = st.session_state.graph.invoke({"messages": [("user", prompt)]}, config)
        state = st.session_state.graph.get_state(config)

        if state.next:
            tool_call = state.values["messages"][-1].tool_calls[0]
            st.session_state.pending_approval = {
                "tool_call_id": tool_call["id"],
                "task": tool_call["args"].get("task", "unknown task"),
            }
            st.rerun()
        else:
            reply = result["messages"][-1].content
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)
    except Exception as e:
        st.error(f"Error communicating with the agent: {e}")
