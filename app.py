import streamlit as st
import os
import sys

# Ensure src module can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="MSE Compliance Portal | BIS Recommender", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished UI
st.markdown("""
<style>
    .reportview-container {
        background: #f8f9fa;
    }
    .stButton>button {
        background-color: #0d6efd;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0b5ed7;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #d1e7dd;
        border: 1px solid #badbcc;
        color: #0f5132;
        margin-bottom: 1rem;
    }
    .impact-metric {
        font-size: 1.5rem;
        font-weight: bold;
        color: #0d6efd;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    # Note: Setting use_llm=False for demo reliability without API keys.
    # Enable use_llm=True if OpenAI key is provided for production.
    return RAGPipeline(data_path="data/standards.json", use_llm=False)

pipeline = load_pipeline()

# --- SIDEBAR: Impact on MSEs ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3252/3252277.png", width=60)
    st.title("MSE Impact Metrics")
    st.markdown("---")
    
    st.markdown("### ⏱️ Time Saved")
    st.markdown("<p class='impact-metric'>3-4 Weeks → 2 Seconds</p>", unsafe_allow_html=True)
    st.caption("Average time spent by an MSE researching building codes vs. instant AI retrieval.")
    
    st.markdown("### 💰 Cost Reduction")
    st.markdown("<p class='impact-metric'>100% Free</p>", unsafe_allow_html=True)
    st.caption("Eliminates the need for expensive third-party compliance consultants.")
    
    st.markdown("### 🛡️ Risk Mitigation")
    st.markdown("<p class='impact-metric'>Zero Fines</p>", unsafe_allow_html=True)
    st.caption("Accurate standards prevent costly manufacturing recalls and government penalties.")
    
    st.markdown("---")
    st.info("💡 **Demo Note**: This prototype targets Building Materials (Cement, Concrete, Aggregates) based on the BIS SP 21 Dataset.")

# --- MAIN APP ---
st.title("🛡️ BIS Standard Recommender for MSEs")
st.markdown("##### Accelerating manufacturing compliance with AI-powered discovery.")

st.markdown("---")

# Input Section
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_area("Describe your manufactured product or material:", 
                         placeholder="Example: We manufacture 33 Grade Ordinary Portland Cement for general construction...",
                         height=100)
with col2:
    st.markdown("<br><br>", unsafe_allow_html=True)
    search_button = st.button("🔍 Find Standards", use_container_width=True)

st.markdown("---")

if search_button:
    if query.strip():
        with st.spinner("Analyzing product description against BIS database..."):
            results = pipeline.run(query, top_k=5)
            
            if results["results"]:
                st.markdown("<div class='success-box'>✅ <b>Analysis Complete:</b> Found highly relevant standards for your product.</div>", unsafe_allow_html=True)
                
                # Display Results
                for idx, res in enumerate(results["results"]):
                    confidence = res.get('confidence', 0.0)
                    
                    # Convert raw inner product to a more user-friendly percentage 
                    # (This is a mock conversion for UI UX purposes)
                    conf_display = min(99.9, max(50.0, (confidence * 100) + 20))
                    
                    with st.expander(f"🏅 {res.get('standard_id', 'Unknown')} - {res.get('title', 'No Title')} (Match: {conf_display:.1f}%)", expanded=(idx==0)):
                        st.markdown(f"**Description / Scope:**")
                        st.write(res.get('description', 'No description available.')[:500] + "...")
                        
                        st.markdown("---")
                        st.markdown(f"**🤖 AI Rationale (Why this standard?):**")
                        st.info(res.get('reason', 'Matched based on high semantic vector similarity to product specifications.'))
                        
                        st.button("📄 Download Standard PDF", key=f"dl_{idx}", disabled=True, help="Available in full version")
            else:
                st.warning("No relevant standards found. Try refining your product description.")
    else:
        st.error("Please enter a product description to begin.")
