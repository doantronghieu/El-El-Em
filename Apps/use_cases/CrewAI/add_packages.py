import sys
import os
from pprint import pprint

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

while True:
    # print(parent_directory)
    sys.path.append(parent_directory)

    parent_folder = parent_directory.split("/")[-1]
    if parent_folder == "Apps":
        break

    parent_directory = os.path.dirname(parent_directory)

# pprint(sys.path)
