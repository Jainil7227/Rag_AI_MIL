import streamlit as st
import sys
import os
import requests
from pathlib import Path
from bs4 import BeautifulSoup

# Add project root to sys.path so package imports like DAY_2.knowledge_base work
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from DAY_3.rag_agent import RAGAgent
    from DAY_2.knowledge_base import KnowledgeBase
except ModuleNotFoundError:
    sys.path.insert(0, os.path.join(project_root, 'DAY_2'))
    sys.path.insert(0, os.path.join(project_root, 'DAY_3'))
    from rag_agent import RAGAgent
    from knowledge_base import KnowledgeBase


def init_session_state():
    """Initialize Streamlit session state."""
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'kb' not in st.session_state:
        st.session_state.kb = None
    
    if 'auto_initialized' not in st.session_state:
        st.session_state.auto_initialized = False


def main():
    st.set_page_config(
        page_title="GDG Knowledge Agent",
        page_icon="https://res.cloudinary.com/startup-grind/image/upload/c_fill,dpr_2.0,f_auto,g_center,h_1200,q_100,w_1200/v1/gcs/platform-data-goog/contentbuilder/GDG_Bevy_SocialSharingThumbnail_KFxxrrs.png",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    
    # Auto-initialize if API key is in environment
    env_api_key = os.getenv('GEMINI_API_KEY')
    if env_api_key and not st.session_state.auto_initialized and st.session_state.agent is None:
        try:
            st.session_state.kb = KnowledgeBase("gdg_knowledge_v2")
            
            # Load GDG guidelines from file if available
            guidelines_path = os.path.join(project_root, 'DAY_2', 'data', 'gdg_guidelines.txt')
            if os.path.exists(guidelines_path):
                with open(guidelines_path, 'r', encoding='utf-8') as f:
                    guidelines_data = f.read()
                
                st.session_state.kb.add_document(
                    guidelines_data,
                    metadata={'source': 'GDG Guidelines', 'type': 'official', 'filename': 'gdg_guidelines.txt'}
                )
            
            st.session_state.agent = RAGAgent(
                gemini_api_key=env_api_key,
                knowledge_base=st.session_state.kb,
            )
            
            st.session_state.auto_initialized = True
            
        except Exception:
            pass
    
    st.title("GDG Knowledge Agent")
    st.markdown("*Powered by Retrieval-Augmented Generation (RAG) with Gemini AI*")
    st.markdown("---")
    
    with st.sidebar:
        env_api_key = os.getenv('GEMINI_API_KEY')
        
        if not env_api_key:
            st.header("⚙️ Configuration")
            
            api_key = st.text_input(
                "Gemini API Key",
                type="password",
                help="Get your free key from https://aistudio.google.com/app/apikey",
                placeholder="Enter your API key here..."
            )
            
            if st.button("🚀 Initialize Agent", type="primary", use_container_width=True):
                if not api_key:
                    st.error("Please provide your Gemini API key!")
                else:
                    with st.spinner("Initializing RAG Agent..."):
                        try:
                            st.session_state.kb = KnowledgeBase("gdg_knowledge_v2")
                            # Load guidelines logic not duplicated here for brevity/cleanliness in sidebar manual init 
                            # (or logic could be extracted to helper function)
                            # For parity with auto-init, keeping it simple:
                            
                            st.session_state.agent = RAGAgent(
                                gemini_api_key=api_key,
                                knowledge_base=st.session_state.kb,
                            )
                            st.success("Agent initialized successfully!")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            
            st.markdown("---")
        
        st.header("📄 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Upload documents to expand knowledge base",
            accept_multiple_files=True,
            type=["txt", "md"],
            help="Upload .txt or .md files"
        )
        
        if uploaded_files and st.button("Process Documents", use_container_width=True):
            if st.session_state.kb is None:
                st.error("Please initialize the agent first!")
            else:
                with st.spinner(f"Processing {len(uploaded_files)} files..."):
                    try:
                        for file in uploaded_files:
                            file_ext = Path(file.name).suffix.lower()
                            text = file.read().decode("utf-8")
                            
                            st.session_state.kb.add_document(
                                text,
                                metadata={'source': file.name, 'type': 'user-uploaded', 'file_type': file_ext}
                            )
                        st.success(f"Processed {len(uploaded_files)} documents successfully!")
                    except Exception as e:
                        st.error(f"Error processing files: {str(e)}")
        
        st.markdown("---")
        
        st.header("🌐 Fetch Live Data")
        
        gdg_url = st.text_input(
            "GDG Chapter URL",
            placeholder="https://gdg.community.dev/your-chapter/"
        )
        
        if st.button("Fetch Latest Events", use_container_width=True):
            if st.session_state.kb is None:
                st.error("Please initialize the agent first!")
            elif not gdg_url:
                st.warning("Please enter a GDG chapter URL!")
            else:
                with st.spinner(f"Fetching data from {gdg_url}..."):
                    try:
                        headers = {'User-Agent': 'Mozilla/5.0'}
                        response = requests.get(gdg_url, headers=headers, timeout=10)
                        response.raise_for_status()
                        
                        soup = BeautifulSoup(response.text, 'html.parser')
                        for script in soup(["script", "style"]):
                            script.decompose()
                        
                        text = soup.get_text()
                        lines = (line.strip() for line in text.splitlines())
                        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                        text = '\n'.join(chunk for chunk in chunks if chunk)
                        
                        st.session_state.kb.add_document(
                            text,
                            metadata={
                                'source': gdg_url,
                                'type': 'web-scraped',
                                'fetched_at': str(os.popen('echo %date% %time%').read().strip())
                            }
                        )
                        st.success("Successfully fetched data!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        st.markdown("---")
        
        if st.session_state.kb:
            st.header("📊 Knowledge Base Stats")
            stats = st.session_state.kb.get_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Chunks", stats['total_chunks'])
            with col2:
                st.metric("Embedding Dim", stats['embedding_dimension'])
            
            if st.button("🔄 Reset Knowledge Base", use_container_width=True, type="secondary"):
                st.session_state.agent = None
                st.session_state.kb = None
                st.session_state.messages = []
                st.session_state.auto_initialized = False
                st.success("Knowledge base reset!")
                st.rerun()
    
    if st.session_state.agent is None:
        env_api_key = os.getenv('GEMINI_API_KEY')
        if env_api_key:
            st.info("🔄 Initializing agent...")
        else:
            st.info("👈 Please configure and initialize the agent in the sidebar to begin")
            st.markdown("## Ask Questions about GDG here once the agent is ready!")
    else:
        st.header("💬 Ask Me Anything!")
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "sources" in message and message["sources"]:
                    with st.expander(f"📚 View {len(message['sources'])} Sources"):
                        for i, source in enumerate(message['sources'], 1):
                            st.markdown(f"**Source {i}:** {source['metadata'].get('source', 'Unknown')}")
                            st.text(source['text'][:200] + "...")
                            st.markdown("---")
        
        if prompt := st.chat_input("Ask about GDG events, workshops..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = st.session_state.agent.answer(prompt, verbose=False)
                        st.markdown(result['answer'])
                        
                        if result['sources']:
                            with st.expander(f"📚 View {len(result['sources'])} Sources"):
                                for i, source in enumerate(result['sources'], 1):
                                    st.markdown(f"**Source {i}:** {source['metadata'].get('source', 'Unknown')}")
                                    st.text(source['text'][:200] + "...")
                                    st.markdown("---")
                        else:
                            st.caption("No sources found in knowledge base")
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result['answer'],
                            "sources": result['sources']
                        })
                    except Exception as e:
                        st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()