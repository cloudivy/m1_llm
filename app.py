import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO
from groq import Groq
import time

# Configure matplotlib
plt.rcParams['figure.max_open_warning'] = 50

st.set_page_config(
    page_title="Leak Detection Dashboard", 
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Pipeline Leak Detection Dashboard")
st.markdown("**AI-Powered Analysis for Quick Leak Location Detection**")

# --- Sidebar: Groq API Configuration ---
with st.sidebar:
    st.header("🤖 AI Configuration")
    st.markdown("### Groq API Key")
    api_key = st.text_input(
        "Enter Groq API Key:", 
        type="password",
        help="Get free key: https://console.groq.com/keys"
    )
    
    if st.button("🔗 Test Connection", type="secondary", use_container_width=True):
        if api_key:
            try:
                client = Groq(api_key=api_key)
                response = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": "Connected!"}],
                    max_tokens=5
                )
                st.success("✅ **Connected successfully!**")
                st.session_state.groq_client = client
                st.session_state.api_key_valid = True
            except Exception as e:
                st.error(f"❌ **Connection failed:** {str(e)[:100]}...")
        else:
            st.warning("👆 Enter API key first")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'dataframes' not in st.session_state:
    st.session_state.dataframes = {}
if 'api_key_valid' not in st.session_state:
    st.session_state.api_key_valid = False

# --- Main Tabs ---
tab1, tab2, tab3 = st.tabs(["📁 Data Upload", "📊 Visual Analysis", "🤖 AI Insights"])

# Tab 1: Data Upload
with tab1:
    st.header("Upload Pipeline Data")
    
    uploaded_files = st.file_uploader(
        "Choose Excel files:",
        type=['xlsx'], 
        accept_multiple_files=True,
        help="Required: df_manual_digging.xlsx, df_lds_IV.xlsx"
    )
    
    if uploaded_files:
        file_dict = {f.name: f for f in uploaded_files}
        st.success(f"✅ Loaded **{len(uploaded_files)}** files")
        
        try:
            # Load dataframes
            if 'df_pidws.xlsx' in file_dict:
                st.session_state.dataframes['df_pidws'] = pd.read_excel(file_dict['df_pidws.xlsx'])
            if 'df_manual_digging.xlsx' in file_dict:
                st.session_state.dataframes['df_manual_digging'] = pd.read_excel(file_dict['df_manual_digging.xlsx'])
            if 'df_lds_IV.xlsx' in file_dict:
                st.session_state.dataframes['df_lds_IV'] = pd.read_excel(file_dict['df_lds_IV.xlsx'])
            
            # Metrics
            digging_count = len(st.session_state.dataframes.get('df_manual_digging', pd.DataFrame()))
            leaks_count = len(st.session_state.dataframes.get('df_lds_IV', pd.DataFrame()))
            pidws_count = len(st.session_state.dataframes.get('df_pidws', pd.DataFrame()))
            
            col1, col2, col3 = st.columns(3)
            col1.metric("🔵 Digging Events", f"{digging_count:,}")
            col2.metric("🔴 Leak Events", f"{leaks_count:,}")
            col3.metric("📊 PIDWS Data", f"{pidws_count:,}")
            
        except Exception as e:
            st.error(f"❌ File error: {str(e)}")

# Tab 2: Visual Analysis
with tab2:
    if not st.session_state.dataframes:
        st.info("👆 **Upload data first** in the Data tab")
        st.stop()
    
    df_digging = st.session_state.dataframes.get('df_manual_digging', pd.DataFrame())
    df_leaks = st.session_state.dataframes.get('df_lds_IV', pd.DataFrame())
    
    # Preprocess data
    if not df_digging.empty and 'DateTime' in df_digging.columns:
        df_digging['Date_new'] = pd.to_datetime(df_digging['DateTime']).dt.date
        df_digging['Time_new'] = pd.to_datetime(df_digging['DateTime']).dt.time
    
    if not df_leaks.empty:
        df_leaks['Date'] = pd.to_datetime(df_leaks['Date'])
        df_leaks['Time'] = pd.to_timedelta(df_leaks['Time'].astype(str))
        df_leaks['DateTime'] = df_leaks['Date'] + df_leaks['Time']
    
    st.success("✅ **Data ready for analysis**")
    
    # Analysis controls
    col1, col2 = st.columns(2)
    with col1:
        target_chainage = st.number_input("🎯 Target Chainage (km)", 25.4, 0.0, 100.0, 0.1)
    with col2:
        tolerance = st.number_input("📏 Tolerance (±km)", 1.0, 0.1, 5.0, 0.1)
    
    if st.button("🚀 **Analyze Chainage**", type="primary", use_container_width=True):
        # Filter data
        digging_filtered = df_digging[
            abs(df_digging['Original_chainage'] - target_chainage) <= tolerance
        ] if not df_digging.empty else pd.DataFrame()
        
        leaks_filtered = df_leaks[
            abs(df_leaks['chainage'] - target_chainage) <= tolerance
        ] if not df_leaks.empty else pd.DataFrame()
        
        # Results
        col1, col2, col3 = st.columns(3)
        col1.metric("🔵 Digging Events", len(digging_filtered))
        col2.metric("🔴 Leak Events", len(leaks_filtered))
        col3.metric("📈 Correlation", f"{len(digging_filtered) + len(leaks_filtered)} total")
        
        # Visualization
        plt.close('all')
        fig, ax = plt.subplots(figsize=(16, 9))
        
        if not digging_filtered.empty:
            sns.scatterplot(
                data=digging_filtered, x='DateTime', y='Original_chainage',
                color='blue', label='Manual Digging', marker='o', s=80, ax=ax
            )
        
        if not leaks_filtered.empty:
            sns.scatterplot(
                data=leaks_filtered, x='DateTime', y='chainage',
                color='red', label='Detected Leaks', marker='X', s=120, ax=ax
            )
        
        plt.title(f'🔍 Chainage {target_chainage} ±{tolerance}km: Digging vs Leaks', 
                 fontsize=16, pad=20)
        plt.xlabel('Date & Time', fontsize=12)
        plt.ylabel('Chainage (km)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)

# Tab 3: AI Chat
with tab3:
    st.header("🤖 AI Pipeline Analyst")
    
    if not st.session_state.dataframes:
        st.warning("👆 **Upload data first**")
    elif not st.session_state.api_key_valid:
        st.warning("👆 **Enter valid Groq API key** in sidebar")
    else:
        # Data context for AI
        df_digging = st.session_state.dataframes.get('df_manual_digging', pd.DataFrame())
        df_leaks = st.session_state.dataframes.get('df_lds_IV', pd.DataFrame())
        
        context = f"""
        **Pipeline Data Summary:**
        • Manual digging events: {len(df_digging)}
        • Leak detection events: {len(df_leaks)}
        • Digging columns: {', '.join(df_digging.columns[:5])}
        • Leaks columns: {', '.join(df_leaks.columns[:5])}
        
        Analyze correlations, predict leak patterns, recommend detection improvements.
        """
        st.info(context)
        
        # Chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # Chat input
        if prompt := st.chat_input("💭 Ask about leaks, chainages, patterns..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                st.markdown("**🤔 Analyzing...**")
                
                client = st.session_state.groq_client
                ai_context = f"{context}\n\n**User Question:** {prompt}"
                
                response = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[
                        {"role": "system", "content": "You are a pipeline leak detection expert working for Indian Oil Corporation. Provide specific, actionable insights for operations managers."},
                        {"role": "user", "content": ai_context}
                    ],
                    temperature=0.1,
                    max_tokens=1200
                )
                
                answer = response.choices[0].message.content
                st.markdown(answer)
                
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

# Footer
st.markdown("---")
st.markdown("""
*Built for pipeline operations at Indian Oil Corporation Limited | 🔍 Quick leak location analytics*
""")
