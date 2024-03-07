import sys
import os
from pprint import pprint

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
p_parent_directory = os.path.dirname(parent_directory)

sys.path.append(p_parent_directory)

# optional
p_p_parent_directory = os.path.dirname(p_parent_directory)
sys.path.append(p_p_parent_directory)
