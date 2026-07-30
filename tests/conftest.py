"""讓 tests/ 底下的測試能 import 到套件根的 models/ 與 utils/。"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # regional_model/DLAMPty_polar
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
