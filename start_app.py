import subprocess
import os
import sys
import platform

def start_services():
    """Starts the Flask backend and React frontend in new console windows."""
    
    current_dir = os.getcwd()
    frontend_dir = os.path.join(current_dir, "autism_web", "autism-website")
    
    print("🚀 Starting Autism Screening Agent Services...")
    
    # Check OS to determine how to open new terminal windows
    system = platform.system()
    
    # Commands
    backend_cmd = f"python app.py"
    frontend_cmd = f"npm start"
    
    if system == "Windows":
        # Launch Backend in new window
        print("🔹 Launching Backend (Flask)...")
        subprocess.Popen(f'start cmd /k "title Autism Backend & {backend_cmd}"', shell=True)
        
        # Launch Frontend in new window
        print("🔹 Launching Frontend (React)...")
        subprocess.Popen(f'start cmd /k "cd /d {frontend_dir} & title Autism Frontend & {frontend_cmd}"', shell=True)
        
    else:
        # Fallback for Linux/Mac (simplified, might need adjustments for specific terminals like gnome-terminal)
        print("⚠️  This script is optimized for Windows. On Linux/Mac, use separate terminals.")
        print(f"1. Run: {backend_cmd}")
        print(f"2. Run: cd {frontend_dir} && {frontend_cmd}")
        return

    print("\n✅ Services launched! You can close this window if you like.")

if __name__ == "__main__":
    try:
        start_services()
    except Exception as e:
        print(f"❌ Error starting services: {e}")
        input("Press Enter to exit...")
