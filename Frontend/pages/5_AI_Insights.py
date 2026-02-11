"""AI Insights Page - Chat with AI about your data."""

import streamlit as st
import requests
import time

st.set_page_config(
    page_title="AI Insights - AutoML Advisor",
    page_icon="🤖",
    layout="wide"
)

API_URL = "http://localhost:8000/api"

# Check for session
if "session_id" not in st.session_state:
    st.warning("⚠️ No analysis session found. Please upload data first.")
    if st.button("📤 Go to Upload"):
        st.switch_page("pages/1_Upload.py")
    st.stop()

session_id = st.session_state.session_id

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Header
st.title("🤖 AI Insights & Chat")
st.markdown("*Ask questions about your churn analysis and get AI-powered insights*")

# Check LLM status
@st.cache_data(ttl=30)
def check_llm_status():
    try:
        response = requests.get(f"{API_URL}/llm/status", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {"available": False, "provider": "unknown", "error": "Could not connect"}

llm_status = check_llm_status()

# Status indicator
col1, col2 = st.columns([3, 1])
with col2:
    if llm_status.get("available"):
        st.success(f"🟢 AI Online ({llm_status.get('provider', 'ollama')})")
    else:
        st.error("🔴 AI Offline")
        st.markdown("""
        **To enable AI features:**
        1. Install [Ollama](https://ollama.com)
        2. Run: `ollama pull llama3.2`
        3. Start Ollama
        """)

st.markdown("---")

# Two columns: AI Summary + Chat
left_col, right_col = st.columns([1, 1])

with left_col:
    st.markdown("### 💡 AI-Generated Summary")
    
    if st.button("✨ Generate AI Insights", use_container_width=True, type="primary"):
        if not llm_status.get("available"):
            st.error("AI is not available. Please start Ollama.")
        else:
            with st.spinner("🤔 AI is analyzing your data..."):
                try:
                    response = requests.post(
                        f"{API_URL}/insights/generate",
                        json={"session_id": session_id},
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.ai_insight = data.get("insight", "")
                    else:
                        st.error("Failed to generate insights")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Display AI insight
    if "ai_insight" in st.session_state and st.session_state.ai_insight:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea22, #764ba222); 
                    padding: 20px; border-radius: 10px; border: 1px solid #667eea44;">
            {st.session_state.ai_insight}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Click 'Generate AI Insights' to get an AI-powered summary of your churn analysis.")

with right_col:
    st.markdown("### 💬 Ask Questions")
    
    # Chat container
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div style="background: #3498db22; padding: 10px 15px; border-radius: 15px 15px 5px 15px; 
                            margin: 5px 0 5px 50px; border: 1px solid #3498db44;">
                    <strong>You:</strong> {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: #2ecc7122; padding: 10px 15px; border-radius: 15px 15px 15px 5px; 
                            margin: 5px 50px 5px 0; border: 1px solid #2ecc7144;">
                    <strong>AI:</strong> {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
    
    # Input
    user_input = st.text_input(
        "Ask about your data...",
        placeholder="e.g., Which customers should I focus on first?",
        key="chat_input",
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        send_button = st.button("Send", use_container_width=True, type="primary")
    with col2:
        if st.button("Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    if send_button and user_input:
        if not llm_status.get("available"):
            st.error("AI is not available. Please start Ollama.")
        else:
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Get AI response
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{API_URL}/chat",
                        json={"session_id": session_id, "message": user_input},
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        ai_response = data.get("response", "Sorry, I couldn't generate a response.")
                        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                    else:
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "content": "Sorry, there was an error processing your question."
                        })
                except Exception as e:
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": f"Error: {str(e)}"
                    })
            
            st.rerun()

st.markdown("---")

# Quick question suggestions
st.markdown("### 💭 Suggested Questions")

suggestions = [
    "What's the most important action I should take today?",
    "Why are my high-risk customers leaving?",
    "How much revenue can I save with a retention campaign?",
    "Which customer segment needs the most attention?",
    "What discount should I offer to at-risk customers?",
    "How do I reduce my churn rate by 10%?"
]

cols = st.columns(3)
for i, suggestion in enumerate(suggestions):
    with cols[i % 3]:
        if st.button(f"💬 {suggestion}", key=f"sug_{i}", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": suggestion})
            
            if llm_status.get("available"):
                try:
                    response = requests.post(
                        f"{API_URL}/chat",
                        json={"session_id": session_id, "message": suggestion},
                        timeout=60
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "content": data.get("response", "")
                        })
                except:
                    pass
            
            st.rerun()

# Sidebar
with st.sidebar:
    st.markdown("### 🤖 AI Status")
    
    if llm_status.get("available"):
        st.success("AI is ready!")
        st.markdown(f"**Provider:** {llm_status.get('provider', 'Ollama')}")
        if llm_status.get("models"):
            st.markdown(f"**Models:** {', '.join(llm_status['models'][:3])}")
    else:
        st.error("AI is offline")
        st.markdown("""
        **Setup Instructions:**
        1. Download Ollama from [ollama.com](https://ollama.com)
        2. Install and run it
        3. Open terminal and run:
        ```
        ollama pull llama3.2
        ```
        4. Refresh this page
        """)
    
    st.divider()
    
    if st.button("🔄 Refresh AI Status", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.divider()
    
    st.markdown("### 📊 Navigation")
    if st.button("📊 Dashboard", use_container_width=True):
        st.switch_page("pages/2_Dashboard.py")
    if st.button("🔍 Customer Explorer", use_container_width=True):
        st.switch_page("pages/3_Explorer.py")
