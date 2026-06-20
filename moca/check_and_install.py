import subprocess
import sys


required_packages = [
    "joblib",
    "matplotlib",
    "pandas",
    "seaborn",
    "scikit-learn",
    "openpyxl",
    "customtkinter"
]


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"{package} not found. Installing...")
        install(package)
