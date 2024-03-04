import sys
import os

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
p_parent_directory = os.path.dirname(parent_directory)

sys.path.append(p_parent_directory)
