import subprocess
import os
import sys

if __name__ == "__main__":
    # Ensure we run from the root directory but point uvicorn to the backend app
    root_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(root_dir, "backend")
    
    print("="*60)
    print("🚀 Starting HR Attrition Predictor Server")
    print("📍 Running on: http://127.0.0.1:8000")
    print("="*60)
    
    # Run uvicorn via subprocess to perfectly emulate running it from the command line in the backend folder
    try:
        subprocess.run(
            [sys.executable, "-m", "uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
            cwd=backend_dir,
            check=True
        )
    except KeyboardInterrupt:
        print("\nServer stopped.")
