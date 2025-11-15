@echo off
echo Locating project folder...
:: This command changes the directory to where the .bat file is
cd /d %~dp0

echo Activating virtual environment...
:: This calls the activation script for your venv
call venv\Scripts\activate.bat

echo Launching LORA Dataset Builder App...
echo This may take a moment to load the model.
echo.
echo Your app will be running at: http://127.0.0.1:8123
echo (Or whatever port you set in app.py)
echo.
echo Press Ctrl+C in this window to stop the server.
echo.
set CUDA_VISIBLE_DEVICES=0
:: This runs your Gradio app
python app.py

:: This keeps the window open after the script finishes or crashes
pause