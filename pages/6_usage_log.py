"""
Usage Log page - Admin-only view of who used the system and what they queried.
Requires a separate ADMIN_PASSWORD secret.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import streamlit as st
import pandas as pd

from src.utils.usage_logger import read_usage_log
from src.utils.auth import require_auth

require_auth()


def _get_secret(key, default=""):
    val = os.environ.get(key, "")
    if not val:
        try:
            val = st.secrets.get(key, "")
        except Exception:
            val = ""
    return val or default


# --- Admin gate ---
def check_admin():
    admin_pw = _get_secret("ADMIN_PASSWORD")
    if not admin_pw:
        st.error("ADMIN_PASSWORD secret is not configured. Usage log is disabled.")
        return False

    if st.session_state.get("admin_authenticated"):
        return True

    st.markdown("## Usage Log (Admin)")
    st.markdown("This page is restricted to the system administrator.")
    pw = st.text_input("Admin password", type="password", key="admin_pw_input")
    if st.button("Unlock", type="primary"):
        if pw == admin_pw:
            st.session_state["admin_authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect admin password.")
    return False


if not check_admin():
    st.stop()

# --- Log viewer (only after admin auth) ---

st.markdown("## Usage Log")
st.markdown("Record of queries submitted to the system.")

entries = read_usage_log()

if not entries:
    st.info("No usage recorded yet.")
    st.stop()

# Convert to DataFrame
df = pd.DataFrame(entries)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp", ascending=False)

# Summary stats
st.markdown(f"**{len(df)} queries** by **{df['user'].nunique()} user(s)**")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total queries", len(df))
with col2:
    st.metric("Unique users", df["user"].nunique())
with col3:
    page_counts = df["page"].value_counts()
    most_used = page_counts.index[0] if len(page_counts) > 0 else "N/A"
    st.metric("Most used page", most_used)

st.markdown("---")

# Filters
with st.sidebar:
    st.markdown("### Filters")

    users = ["All"] + sorted(df["user"].unique().tolist())
    selected_user = st.selectbox("User", users)

    pages = ["All"] + sorted(df["page"].unique().tolist())
    selected_page = st.selectbox("Page", pages)

# Apply filters
filtered = df.copy()
if selected_user != "All":
    filtered = filtered[filtered["user"] == selected_user]
if selected_page != "All":
    filtered = filtered[filtered["page"] == selected_page]

st.markdown(f"Showing **{len(filtered)}** of {len(df)} entries")

# Display table
display_df = filtered[["timestamp", "user", "page", "query"]].copy()
display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
display_df.columns = ["Time", "User", "Page", "Query"]
st.dataframe(display_df, use_container_width=True, hide_index=True)

# Export
st.markdown("---")
log_json = json.dumps(entries, indent=2, ensure_ascii=False)
st.download_button(
    label="Download full log (JSON)",
    data=log_json,
    file_name="usage_log.json",
    mime="application/json",
)
