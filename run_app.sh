#!/bin/bash
echo "Locating project folder..."
# This command changes the directory to where the .sh file is
cd "$(dirname "$0")"

echo "Activating virtual environment..."
# This calls the activation script for your venv
source venv/bin/activate

echo "Launching LORA Dataset Builder App..."
echo "This may take a moment to load the model."
echo ""
echo "Your app will be running at: http://127.0.0.1:8123"
echo "(Or whatever port you set in app.py)"
echo ""
echo "Press Ctrl+C in this window to stop the server."
echo ""
# This runs your Gradio app
python app.py

# Keep the window open after script finishes/crashes (optional)
read -p "Press Enter to close..."