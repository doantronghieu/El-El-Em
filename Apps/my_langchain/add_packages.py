import sys
import os
from pprint import pprint

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

while os.path.basename(parent_directory) != "Apps":
    sys.path.append(parent_directory)
    parent_directory = os.path.dirname(parent_directory)
    print(parent_directory)

# pprint(sys.path)
