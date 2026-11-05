from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from agents.workflow import workflow
from agents.analyst import answer_question


# -----------------------------------------------------------------------------
# Page Configuration & Dark Dashboard Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Data Analysis Agent | Executive Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* Base Dark Theme Overrides */
.stApp {
    background: radial-gradient(circle at 18% 12%, #141e33 0%, #090d16 100%) !important;
    color: #e2e8f0 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Streamlit Header Bar Fix */
header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background: #0d1322 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 3.5rem !important;
    padding-left: 1.1rem !important;
    padding-right: 1.1rem !important;
}
.sidebar-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 1.15rem;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 0.25rem;
}
.sidebar-subtitle {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-bottom: 1.25rem;
}
.sidebar-section-title {
    font-size: 0.8rem;
    font-weight: 700;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 1.1rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 6px;
}
.sidebar-card {
    background: rgba(22, 33, 56, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 0.75rem 0.9rem;
    margin-bottom: 0.75rem;
    font-size: 0.85rem;
}

/* Main Container: Generous top padding so content is never clipped by top bar */
.block-container {
    padding-top: 5.25rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1280px !important;
}

/* Header Badge & Hero */
.hero-badge {
    margin-top: 0.5rem;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: 9999px;
    padding: 4px 14px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #38bdf8;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}
.hero-badge-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background-color: #38bdf8;
    box-shadow: 0 0 10px #38bdf8;
}
.hero-title {
    font-size: 2.35rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    line-height: 1.15;
    background: linear-gradient(90deg, #ffffff 0%, #93c5fd 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 400;
    line-height: 1.5;
    margin-bottom: 1.5rem;
}

/* Prominent Upload Hero Area */
.upload-card {
    background: linear-gradient(145deg, rgba(22, 33, 58, 0.75) 0%, rgba(13, 20, 36, 0.85) 100%);
    border: 1.5px dashed rgba(99, 102, 241, 0.45);
    border-radius: 18px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.75rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    transition: all 0.3s ease;
}
.upload-card:hover {
    border-color: rgba(56, 189, 248, 0.7);
    box-shadow: 0 8px 36px rgba(56, 189, 248, 0.15);
}
.upload-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 0.8rem;
}
.upload-icon {
    font-size: 1.6rem;
    background: rgba(99, 102, 241, 0.18);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 12px;
    width: 44px;
    height: 44px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.upload-title {
    color: #f1f5f9;
    font-size: 1.15rem;
    font-weight: 700;
    margin: 0;
}
.upload-desc {
    color: #94a3b8;
    font-size: 0.88rem;
    margin: 0;
}

/* Active File Banner */
.active-file-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.25);
    border-radius: 12px;
    padding: 0.85rem 1.25rem;
    margin-bottom: 1.5rem;
}
.active-file-title {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #34d399;
    font-weight: 600;
    font-size: 0.95rem;
}
.active-file-badge {
    background: rgba(16, 185, 129, 0.2);
    color: #6ee7b7;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 3px 10px;
    border-radius: 9999px;
    text-transform: uppercase;
}

/* KPI Summary Cards */
.kpi-box {
    background: linear-gradient(135deg, rgba(24, 34, 58, 0.8) 0%, rgba(14, 21, 38, 0.85) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.kpi-box:hover {
    transform: translateY(-2px);
    border-color: rgba(56, 189, 248, 0.35);
}
.kpi-label {
    color: #94a3b8;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.kpi-val {
    color: #f8fafc;
    font-size: 1.7rem;
    font-weight: 700;
    line-height: 1.1;
}
.kpi-sub {
    font-size: 0.8rem;
    font-weight: 500;
    margin-top: 0.4rem;
}
.kpi-green { color: #34d399; }
.kpi-cyan { color: #38bdf8; }
.kpi-purple { color: #c084fc; }
.kpi-amber { color: #fbbf24; }

/* Polished Tab Bar */
div[data-baseweb="tab-list"] {
    background: rgba(15, 23, 42, 0.75) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    padding: 6px !important;
    gap: 8px !important;
    margin-bottom: 1.5rem !important;
}
div[data-baseweb="tab"] {
    border-radius: 10px !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.2s ease !important;
}
div[data-baseweb="tab"]:hover {
    color: #f1f5f9 !important;
    background: rgba(255, 255, 255, 0.04) !important;
}
div[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.2) 0%, rgba(99, 102, 241, 0.25) 100%) !important;
    color: #38bdf8 !important;
    border: 1px solid rgba(56, 189, 248, 0.35) !important;
    box-shadow: 0 4px 15px rgba(56, 189, 248, 0.15) !important;
}

/* Cleaning Action Item Cards */
.action-card {
    background: rgba(18, 27, 46, 0.75);
    border-left: 4px solid #38bdf8;
    border-radius: 0 10px 10px 0;
    padding: 0.8rem 1.1rem;
    margin-bottom: 0.65rem;
    font-size: 0.92rem;
    color: #e2e8f0;
}

/* Chat Styling */
.chat-hero-banner {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(56, 189, 248, 0.08) 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 14px;
    padding: 1rem 1.3rem;
    margin-bottom: 1.25rem;
}
.stChatMessage {
    background: rgba(18, 26, 45, 0.7) !important;
    border: 1px solid rgba(255, 255, 255, 0.07) !important;
    border-radius: 14px !important;
    padding: 0.9rem 1.2rem !important;
    margin-bottom: 0.9rem !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25) !important;
}
div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
    border-left: 3.5px solid #818cf8 !important;
    background: rgba(28, 38, 64, 0.6) !important;
}
div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
    border-left: 3.5px solid #38bdf8 !important;
    background: rgba(16, 25, 44, 0.75) !important;
}
div[data-testid="stChatInput"] {
    background: rgba(15, 23, 42, 0.9) !important;
    border: 1px solid rgba(99, 102, 241, 0.4) !important;
    border-radius: 14px !important;
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.45) !important;
}

/* Feature Grid */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.25rem;
    margin-top: 1.5rem;
}
.feature-card {
    background: rgba(18, 27, 47, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.4rem;
    transition: all 0.3s ease;
}
.feature-card:hover {
    border-color: rgba(99, 102, 241, 0.4);
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
}
.feature-icon {
    font-size: 1.5rem;
    margin-bottom: 0.6rem;
}
.feature-heading {
    color: #f1f5f9;
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 0.35rem;
}
.feature-text {
    color: #94a3b8;
    font-size: 0.88rem;
    line-height: 1.5;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session State Initialization (Datasets & Multi-Thread Conversations)
# -----------------------------------------------------------------------------
if "uploaded_datasets" not in st.session_state:
    st.session_state.uploaded_datasets = {}

if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None

if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None

if "workflow_result" not in st.session_state:
    st.session_state.workflow_result = None

# Multi-conversation management
if "conversations" not in st.session_state:
    st.session_state.conversations = {"Chat 1": []}

if "active_conversation" not in st.session_state:
    st.session_state.active_conversation = "Chat 1"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# -----------------------------------------------------------------------------
# Sidebar: Uploaded Datasets & Conversation History
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚡ Data Agent Pro</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Autonomous Cleaning & Analytics</div>', unsafe_allow_html=True)

    # 1. Previous Datasets Section
    st.markdown('<div class="sidebar-section-title">📁 Datasets Library</div>', unsafe_allow_html=True)

    if st.session_state.uploaded_datasets:
        dataset_list = list(st.session_state.uploaded_datasets.keys())
        current_idx = (
            dataset_list.index(st.session_state.dataset_name)
            if st.session_state.dataset_name in dataset_list
            else 0
        )

        selected_dataset = st.selectbox(
            "Active Dataset",
            options=dataset_list,
            index=current_idx,
            key="sidebar_dataset_selector",
            help="Switch between previously uploaded datasets.",
        )

        # Handle dataset switching
        if selected_dataset != st.session_state.dataset_name:
            ds_info = st.session_state.uploaded_datasets[selected_dataset]
            st.session_state.dataset_name = selected_dataset
            st.session_state.cleaned_df = ds_info["cleaned_df"]
            st.session_state.workflow_result = ds_info["workflow_result"]
            st.rerun()

        # Display active dataset quick stats
        active_df = st.session_state.cleaned_df
        if active_df is not None:
            st.markdown(
                f"""
                <div class="sidebar-card">
                    <div>📊 <strong>{len(active_df):,}</strong> rows</div>
                    <div>🏷️ <strong>{len(active_df.columns)}</strong> columns</div>
                    <div>✨ Status: <span style="color:#34d399;">Cleaned & Ready</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.caption("No datasets uploaded yet. Upload a CSV to get started.")

    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 1rem 0;'>", unsafe_allow_html=True)

    # 2. Previous Conversations Section
    st.markdown('<div class="sidebar-section-title">💬 Conversations</div>', unsafe_allow_html=True)

    # Button to start a new chat thread
    if st.button("➕ New Chat", use_container_width=True):
        new_chat_num = len(st.session_state.conversations) + 1
        new_chat_id = f"Chat {new_chat_num}"
        st.session_state.conversations[new_chat_id] = []
        st.session_state.active_conversation = new_chat_id
        st.session_state.chat_history = []
        st.rerun()

    convo_list = list(st.session_state.conversations.keys())
    convo_idx = (
        convo_list.index(st.session_state.active_conversation)
        if st.session_state.active_conversation in convo_list
        else 0
    )

    selected_convo = st.radio(
        "Saved Threads",
        options=convo_list,
        index=convo_idx,
        key="sidebar_conversation_radio",
        label_visibility="collapsed",
    )

    if selected_convo != st.session_state.active_conversation:
        st.session_state.active_conversation = selected_convo
        st.session_state.chat_history = st.session_state.conversations[selected_convo]
        st.rerun()

    # Clear conversations button
    if st.button("🗑️ Clear All Conversations", use_container_width=True):
        st.session_state.conversations = {"Chat 1": []}
        st.session_state.active_conversation = "Chat 1"
        st.session_state.chat_history = []
        st.rerun()


# -----------------------------------------------------------------------------
# Main App Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div>
        <div class="hero-badge">
            <span class="hero-badge-dot"></span>
            Autonomous Data Intelligence
        </div>
        <div class="hero-title">Data Analysis Agent</div>
        <div class="hero-subtitle">
            Upload any CSV dataset to execute automated LLM-driven cleaning, full quality profiling,
            and interactive conversational analytics.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Ingestion Area (Upload Logic Preserved & Recorded to Dataset Library)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="upload-card">
        <div class="upload-header">
            <div class="upload-icon">📁</div>
            <div>
                <h3 class="upload-title">Dataset Ingestion</h3>
                <p class="upload-desc">Drop your CSV dataset here to automatically plan, validate, clean, and analyze.</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Select or drop a CSV file",
    type=["csv"],
    help="Supports any standard CSV file with headers.",
    label_visibility="collapsed",
)

if uploaded_file is not None:
    # Process a newly uploaded file only once.
    if st.session_state.dataset_name != uploaded_file.name:
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".csv",
                delete=False,
            ) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                csv_path = temp_file.name

            with st.spinner("Analyzing, planning, and executing dataset cleaning..."):
                result = workflow.invoke(
                    {
                        "csv_path": csv_path,
                        "question": "",
                    }
                )

            cleaned = result.get("cleaned_df")
            st.session_state.workflow_result = result
            st.session_state.cleaned_df = cleaned
            st.session_state.dataset_name = uploaded_file.name

            # Store in dataset history library
            st.session_state.uploaded_datasets[uploaded_file.name] = {
                "cleaned_df": cleaned,
                "workflow_result": result,
            }

            # Reset chat history for newly loaded dataset
            st.session_state.chat_history = []
            st.session_state.conversations[st.session_state.active_conversation] = []
            st.rerun()

        except Exception as error:
            st.error(f"Could not process the CSV: {error}")


result = st.session_state.workflow_result
df = st.session_state.cleaned_df


# -----------------------------------------------------------------------------
# Main Dashboard View (When Dataset is Loaded)
# -----------------------------------------------------------------------------
if result is not None and df is not None:
    # 1. Active File Indicator Banner
    st.markdown(
        f"""
        <div class="active-file-banner">
            <div class="active-file-title">
                <span>⚡</span>
                <span>Active Dataset: <strong>{st.session_state.dataset_name}</strong></span>
            </div>
            <div class="active-file-badge">Cleaned & Analyzed</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Executive KPI Cards
    report_overview = result.get("report_metrics", {}).get("overview", {})
    orig_rows = report_overview.get("orig_rows", len(df))
    clean_rows = len(df)
    row_delta = report_overview.get("row_delta", clean_rows - orig_rows)

    orig_dup = report_overview.get("orig_duplicates", 0)
    clean_dup = report_overview.get("clean_duplicates", 0)

    orig_missing = report_overview.get("orig_total_missing", int(df.isna().sum().sum()))
    clean_missing = report_overview.get("clean_total_missing", int(df.isna().sum().sum()))
    missing_delta = clean_missing - orig_missing

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

    with col_kpi1:
        st.markdown(
            f"""
            <div class="kpi-box">
                <div class="kpi-label">Cleaned Records</div>
                <div class="kpi-val">{clean_rows:,}</div>
                <div class="kpi-sub kpi-cyan">{f"{row_delta:+d} rows" if row_delta != 0 else "Preserved row count"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_kpi2:
        st.markdown(
            f"""
            <div class="kpi-box">
                <div class="kpi-label">Duplicates Removed</div>
                <div class="kpi-val">{orig_dup - clean_dup:,}</div>
                <div class="kpi-sub kpi-green">{clean_dup} remaining in dataset</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_kpi3:
        st.markdown(
            f"""
            <div class="kpi-box">
                <div class="kpi-label">Missing Values Fixed</div>
                <div class="kpi-val">{abs(missing_delta):,}</div>
                <div class="kpi-sub kpi-amber">{clean_missing:,} remaining nulls</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_kpi4:
        st.markdown(
            f"""
            <div class="kpi-box">
                <div class="kpi-label">Features / Columns</div>
                <div class="kpi-val">{len(df.columns)}</div>
                <div class="kpi-sub kpi-purple">{len(numeric_cols)} numeric · {len(categorical_cols)} categorical</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. Main Dashboard Tabs
    tab_data, tab_report, tab_chat = st.tabs(
        ["📊 Cleaned Data", "🛡️ Cleaning & Quality Report", "💬 AI Analyst Chat"]
    )

    # --- TAB 1: Cleaned Data ---
    with tab_data:
        st.subheader("Cleaned Dataset Preview")
        st.caption(f"Displaying {len(df):,} rows and {len(df.columns)} columns ready for analysis or export.")

        st.dataframe(df, use_container_width=True, height=450)

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download Cleaned CSV",
            data=csv_data,
            file_name=f"cleaned_{st.session_state.dataset_name or 'dataset.csv'}",
            mime="text/csv",
        )

    # --- TAB 2: Cleaning Report ---
    with tab_report:
        col_rep1, col_rep2 = st.columns([1.1, 0.9])

        with col_rep1:
            st.subheader("Execution Log")
            cleaning_log = result.get("cleaning_log", [])

            if cleaning_log:
                for item in cleaning_log:
                    action = item.get("action", "")
                    col = item.get("column", "")

                    if action == "remove_duplicates":
                        desc = f"🗑️ **Remove Duplicates**: Removed <strong>{item.get('rows_affected', 0)}</strong> exact duplicate rows"
                    elif action == "convert_to_numeric":
                        desc = f"🔢 **Convert to Numeric**: Converted <code>{col}</code> ({item.get('values_affected', 0)} values numeric, {item.get('values_became_missing', 0)} coerced to null)"
                    elif action == "standardize_case":
                        desc = f"🔤 **Standardize Case**: Formatted text to title case in <code>{col}</code> ({item.get('values_affected', 0)} values updated)"
                    elif action == "trim_whitespace":
                        col_disp = "all text columns" if col == "all_string_columns" else f"<code>{col}</code>"
                        desc = f"✂️ **Trim Whitespace**: Stripped extra spaces across {col_disp} ({item.get('values_affected', 0)} values trimmed)"
                    elif action == "fill_missing":
                        desc = f"🩹 **Fill Missing**: Imputed <code>{col}</code> using strategy <code>{item.get('strategy')}</code> ({item.get('values_affected', 0)} values filled)"
                    else:
                        desc = f"⚡ **{action}**: {item}"

                    st.markdown(f'<div class="action-card">{desc}</div>', unsafe_allow_html=True)
            else:
                st.info("No cleaning actions were necessary or executed.")

        with col_rep2:
            st.subheader("Data Quality Summary")
            rep_overview = result.get("report_metrics", {}).get("overview")
            if rep_overview:
                st.dataframe(
                    pd.DataFrame([
                        {"Metric": "Rows", "Original": rep_overview.get("orig_rows"), "Cleaned": rep_overview.get("clean_rows"), "Delta": rep_overview.get("row_delta")},
                        {"Metric": "Duplicates", "Original": rep_overview.get("orig_duplicates"), "Cleaned": rep_overview.get("clean_duplicates"), "Delta": rep_overview.get("duplicate_delta")},
                        {"Metric": "Missing Values", "Original": rep_overview.get("orig_total_missing"), "Cleaned": rep_overview.get("clean_total_missing"), "Delta": rep_overview.get("missing_delta")},
                    ]),
                    use_container_width=True,
                    hide_index=True,
                )

        # Visual Comparison Chart (if generated)
        chart_path = Path("reports/dataset_comparison_report.png")
        if chart_path.exists():
            st.markdown("---")
            st.subheader("Visual Before & After Quality Audit")
            st.image(str(chart_path), caption="Generated Before/After Comparison Chart", use_container_width=True)

        st.markdown("---")
        with st.expander("🔍 Deep Dataset Analysis Summary (JSON)"):
            st.json(result.get("analysis", {}))

        with st.expander("📋 Full Column Quality Metrics (JSON)"):
            st.json(result.get("report_metrics", {}))

    # --- TAB 3: AI Analyst Chat ---
    with tab_chat:
        st.markdown(
            f"""
            <div class="chat-hero-banner">
                <div style="font-weight:700; color:#38bdf8; font-size:1.1rem; margin-bottom:4px;">
                    💬 Conversational Dataset Analyst &nbsp;•&nbsp; <span style="font-size:0.9rem; color:#a5b4fc;">{st.session_state.active_conversation}</span>
                </div>
                <div style="color:#94a3b8; font-size:0.9rem;">
                    Ask questions about duplicates, features, missing data, averages, or distributions.
                    The analyst evaluates dataset properties and gives verified answers.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick Suggestion Prompts
        st.caption("Suggested inquiries:")
        col_sug1, col_sug2, col_sug3, col_sug4 = st.columns(4)
        with col_sug1:
            st.markdown("💡 *were there any duplicates?*")
        with col_sug2:
            st.markdown("💡 *how many features are there?*")
        with col_sug3:
            st.markdown("💡 *any missing values?*")
        with col_sug4:
            st.markdown("💡 *give me an overview*")

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Render Chat History
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                # Safely render structured result ONLY if it is a valid dict/list
                res = message.get("result")
                if isinstance(res, (dict, list)) and res:
                    with st.expander("📊 View Structured Data", expanded=False):
                        st.json(res)

        # Chat Input Bar (Preserved Chat Logic)
        if prompt := st.chat_input("Ask a question about the dataset (e.g. were there any duplicates?)..."):
            st.session_state.chat_history.append(
                {"role": "user", "content": prompt}
            )

            with st.chat_message("user"):
                st.write(prompt)

            try:
                # Exclude the current prompt so follow-up resolution sees
                # only earlier conversation turns.
                previous_history = st.session_state.chat_history[:-1]

                response = answer_question(
                    df,
                    prompt,
                    chat_history=previous_history,
                )

                answer = response.get("answer", "I could not answer that question.")
                res_data = response.get("result")

                history_item = {"role": "assistant", "content": answer}
                if isinstance(res_data, (dict, list)) and res_data:
                    history_item["result"] = res_data

                st.session_state.chat_history.append(history_item)

                # Keep active conversation thread in sync
                st.session_state.conversations[st.session_state.active_conversation] = (
                    st.session_state.chat_history
                )

                with st.chat_message("assistant"):
                    st.write(answer)
                    if isinstance(res_data, (dict, list)) and res_data:
                        with st.expander("📊 View Structured Data", expanded=False):
                            st.json(res_data)

            except Exception as error:
                st.error(f"Could not answer the question: {error}")

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        col_clear, _ = st.columns([1, 5])
        with col_clear:
            if st.button("🗑️ Clear current chat"):
                st.session_state.chat_history = []
                st.session_state.conversations[st.session_state.active_conversation] = []
                st.rerun()

# -----------------------------------------------------------------------------
# Empty State: When No Dataset is Uploaded Yet
# -----------------------------------------------------------------------------
elif uploaded_file is None and not st.session_state.uploaded_datasets:
    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-card">
                <div class="feature-icon">🧹</div>
                <div class="feature-heading">Autonomous Cleaning</div>
                <div class="feature-text">
                    Automatically unmasks string-formatted numbers, trims whitespace, standardizes casing,
                    eliminates duplicate rows, and imputes missing values with LLM planning.
                </div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <div class="feature-heading">Quality Auditing</div>
                <div class="feature-text">
                    Generates before-and-after audit reports comparing missing rates, feature dtypes,
                    and unique value counts across any dynamic CSV structure.
                </div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">💬</div>
                <div class="feature-heading">Conversational Insights</div>
                <div class="feature-text">
                    Ask natural-language questions directly to your data. The agent runs verifiable Python
                    aggregations and provides narrative summaries.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
