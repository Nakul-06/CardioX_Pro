"""
CardioX Pro - Streamlit Cloud Entrypoint
Runs app_streamlit.py seamlessly on Streamlit Community Cloud.
"""
import runpy

if __name__ == "__main__":
    runpy.run_path("app_streamlit.py", run_name="__main__")
