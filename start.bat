@echo OFF
echo "Starting Flask backend..."
start "Flask Backend" cmd /c "python app.py"

echo "Starting Streamlit frontend..."
start "Streamlit Frontend" cmd /c "streamlit run streamlit_app.py"
