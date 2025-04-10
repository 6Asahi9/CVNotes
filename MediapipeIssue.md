🐍 Python Virtual Environment Setup for Mediapipe (Windows CMD)
1. py -3.10 --version
✅ Check if Python 3.10 is installed

2. Create a Virtual Environment Using Python 3.10

3. Navigate to your project folder (e.g., HandTracking):
cd path\to\your\Folder

4. Run
py -3.10 -m venv venv

5. Activate the virtual environment:
.\venv\Scripts\activate.bat

6. Open VS Code → Press Ctrl + Shift + P → Python: Select Interpreter → choose Python 3.10.xx (venv)

7. VS Code terminal should now show something like:
(venv) C:\Users\Asahi\Desktop\HandTracking>

8. Install packages inside the venv:
pip install mediapipe opencv-python

