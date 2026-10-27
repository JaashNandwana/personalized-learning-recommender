@echo off
echo =======================================================
echo   Starting Personalized Learning Recommendation System
echo   Streamlit Web Dashboard on http://localhost:8501
echo =======================================================
call "%~dp0.venv\Scripts\activate.bat"
streamlit run jaash_rs/app/app.py
