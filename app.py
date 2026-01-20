import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from groq import Groq
import traceback

# Fix matplotlib warnings
plt.rcParams['figure.max_open_warning'] = 0
plt.style.use('default')

st.set_page_config(page_title="Leak Detection Dashboard", layout="wide", page_icon="🔍")

st.title("🔍 Pipeline Leak Detection Dashboard")
st.markdown("**AI Analytics for IOC Pipeline Operations**")

# Initialize session state
@st.cache_data
def init_session():
    return {
        'chat_history': [],
        'dataframes': {},
        'api_key_valid': False
    }

if 'session_init' not in st.session_state:
    st.session_state.session_init = init_session()

# Sidebar - API Key
with st.sidebar:
    st.header("🤖 Groq Setup")
    api_key = st.text_input("🔑 Groq API Key", type="password", 
                           help="https://console.groq.com/keys")
    
    if st.button("✅ Test API", use_container_width=True):
        if api_key and api_key.startswith('gsk_'):
            try:
                client = Groq(api_key=api_key)
                response = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": "OK"}],
                    max_tokens=5
                )
                st.success("✅ **API Connected!**")
                st.session_state.groq_client = client
                st.session_state.api_key_valid = True
            except Exception as e:
                st.error(f"❌ API Error: {str(e)[:100]}")
        else:
            st.error("❌ Invalid key format (should start with 'gsk_')")

# Main Tabs
tab1, tab2, tab3 = st.tabs(["📁 Upload Data", "📊 Analysis", "🤖 AI Chat"])

# TAB 1: Data Upload
with tab1:
    st.header("Upload Excel Files")
    uploaded_files = st.file_uploader(
        "Choose files", type=['xlsx'], accept_multiple_files=True,
        help="df_manual_digging.xlsx, df_lds_IV.xlsx"
    )
    
    if uploaded_files:
        try:
            dataframes = {}
            for file in uploaded_files:
                df = pd.read_excel(file)
                dataframes[file.name] = df
                st.success(f"✅ {file.name}: {len(df)} rows")
            
            st.session_state.dataframes = dataframes
            
            # Show metrics
            digging = dataframes.get('df_manual_digging.xlsx', pd.DataFrame())
            leaks = dataframes.get('df_lds_IV.xlsx', pd.DataFrame())
            
            col1, col2 = st.columns(2)
            col1.metric("🔵 Digging", len(digging))
            col2.metric("🔴 Leaks", len(leaks))
            
        except Exception as e:
            st.error(f"❌ Upload Error: {str(e)}")
            st.info("💡 Expected columns: DateTime, Original_chainage, chainage")

# TAB 2: Analysis (SINGLE TOLERANCE)
with tab2:
    if not st.session_state.dataframes:
        st.info("👆 Upload files first")
        st.stop()
    
    digging = st.session_state.dataframes.get('df_manual_digging.xlsx', pd.DataFrame())
    leaks = st.session_state.dataframes.get('df_lds_IV.xlsx', pd.DataFrame())
    
    # Safe preprocessing
    def safe_preprocess(df, is_digging=False):
        if df.empty:
            return df
        try:
            if is_digging and 'DateTime' in df.columns:
                df['DateTime'] = pd.to_datetime(df['DateTime'])
            elif 'Date' in df.columns and 'Time' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df['Time'] = pd.to_timedelta(df['Time'].astype(str))
                df['DateTime'] = df['Date'] + df['Time']
            return df
        except:
            return df
    
    digging = safe_preprocess(digging, True)
    leaks = safe_preprocess(leaks, False)
    
    # SIMPLIFIED CONTROLS - SINGLE TOLERANCE
    col1, col2 = st.columns([2, 1])  # Wider chainage input
    chainage = col1.number_input("🎯 **Target Chainage (km)**", 0.0, 100.0, 25.4, 0.1)
    tolerance = col2.number_input("📏 **Tolerance (±km)**", 0.1, 5.0, 1.0, 0.1)
    
    st.info(f"🔍 Searching **{chainage} ±{tolerance} km** range")
    
    if st.button("📈 **RUN ANALYSIS**", type="primary", use_container_width=True):
        # Filter with SINGLE tolerance
        dig_filtered = digging[
            (digging['Original_chainage'] >= chainage - tolerance) & 
            (digging['Original_chainage'] <= chainage + tolerance)
        ] if 'Original_chainage' in digging.columns else pd.DataFrame()
        
        leak_filtered = leaks[
            (leaks['chainage'] >= chainage - tolerance) & 
            (leaks['chainage'] <= chainage + tolerance)
        ] if 'chainage' in leaks.columns else pd.DataFrame()
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("🔵 Digging Events", len(dig_filtered))
        col2.metric("🔴 Leak Events", len(leak_filtered))
        col3.metric("📊 Total Matches", len(dig_filtered) + len(leak_filtered))
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 8))
        colors = {'digging': 'blue', 'leak': 'red'}
        markers = {'digging': 'o', 'leak': 'X'}
        sizes = {'digging': 60, 'leak': 100}
        
        if not dig_filtered.empty:
            ax.scatter(dig_filtered.get('DateTime', range(len(dig_filtered))), 
                      dig_filtered['Original_chainage'], 
                      c=colors['digging'], s=sizes['digging'], 
                      label='Manual Digging', marker=markers['digging'])
        
        if not leak_filtered.empty:
            ax.scatter(leak_filtered.get('DateTime', range(len(leak_filtered))), 
                      leak_filtered['chainage'], 
                      c=colors['leak'], s=sizes['leak'], 
                      label='Detected Leaks', marker=markers['leak'])
        
        ax.axhspan(chainage - tolerance, chainage + tolerance, 
                  alpha=0.2, color='green', label=f'Target Range ±{tolerance}km')
        ax.axhline(chainage, color='orange', linestyle='--', linewidth=2, label=f'Target: {chainage}km')
        
        ax.set_title(f'🔍 Chainage Analysis: {chainage} ±{tolerance}km', fontsize=16, pad=20)
        ax.set_xlabel('DateTime / Index', fontsize=12)
        ax.set_ylabel('Chainage (km)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

        # Summary table
        st.subheader("📋 Results Summary")
        summary_data = {
            'Metric': ['Digging Events', 'Leak Events', 'Total', 'Range Searched'],
            'Value': [len(dig_filtered), len(leak_filtered), len(dig_filtered)+len(leak_filtered), f'{chainage} ±{tolerance}km']
        }
        st.table(pd.DataFrame(summary_data))

# TAB 3: AI Chat (unchanged)
with tab3:
    st.header("🤖 Ask AI About Your Data")
    
    if not st.session_state.dataframes:
        st.warning("👆 Upload data first")
    elif not st.session_state.get('api_key_valid', False):
        st.warning("👆 Enter valid API key")
    else:
        # Chat display
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
            
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # Chat input
        prompt = st.chat_input("Ask about patterns, predictions, improvements...")
        if prompt:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("AI analyzing..."):
                    client = st.session_state.groq_client
                    digging = st.session_state.dataframes.get('df_manual_digging.xlsx', pd.DataFrame())
                    leaks = st.session_state.dataframes.get('df_lds_IV.xlsx', pd.DataFrame())
                    
                    context = f"""
                    Pipeline data: {len(digging)} digging events, {len(leaks)} leak events.
                    Columns - Digging: {list(digging.columns)}
                    Leaks: {list(leaks.columns)}
                    Question: {prompt}
                    """
                    
                    try:
                        response = client.chat.completions.create(
                            model="llama3-8b-8192",
                            messages=[
                                {"role": "system", "content": "You are a pipeline expert for Indian Oil. Give specific leak detection insights."},
                                {"role": "user", "content": context}
                            ],
                            max_tokens=800,
                            temperature=0.2
                        )
                        answer = response.choices[0].message.content
                        st.markdown(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"AI Error: {str(e)}")

st.markdown("---")
st.caption("💡 Indian Oil Corporation | Pipeline Leak Analytics | Single Tolerance Mode")
