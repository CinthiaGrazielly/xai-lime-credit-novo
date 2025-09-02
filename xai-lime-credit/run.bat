@echo off
REM Cria ambiente virtual, instala dependências e executa o pipeline completo
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python src\train_and_explain.py
echo.
echo =============================================================
echo FIM! Resultados salvos na pasta "outputs".
echo - graficos_*.png, lime_explanation_*.html, metrics.txt
echo =============================================================
pause
