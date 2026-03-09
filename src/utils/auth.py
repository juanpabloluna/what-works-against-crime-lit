"""Shared authentication gate for all pages."""

import os
import streamlit as st


def _get_secret(key, default=""):
    """Read a secret from env vars or st.secrets."""
    val = os.environ.get(key, "")
    if not val:
        try:
            val = st.secrets.get(key, "")
        except Exception:
            val = ""
    return val or default


def require_auth():
    """Check authentication. Shows login form and calls st.stop() if not authenticated.

    Call this at the top of every page before any content.
    """
    access_password = _get_secret("ACCESS_PASSWORD")
    if not access_password:
        # No password configured — allow access (local dev)
        if "user_name" not in st.session_state:
            st.session_state["user_name"] = "local_dev"
        return

    if st.session_state.get("authenticated"):
        return

    st.markdown("## Criminal Governance Literature Expert")
    st.markdown("This application is password-protected. Enter your name and the access code to continue.")

    user_name = st.text_input("Your name", key="login_name_input",
                              placeholder="e.g. Juan Pablo Luna")
    password = st.text_input("Access code", type="password", key="password_input")
    if st.button("Enter", type="primary"):
        if not user_name.strip():
            st.error("Please enter your name.")
        elif password == access_password:
            st.session_state["authenticated"] = True
            st.session_state["user_name"] = user_name.strip()
            st.rerun()
        else:
            st.error("Incorrect access code.")
    st.stop()
