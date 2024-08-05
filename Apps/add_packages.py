import sys
import os
from pprint import pprint
from dotenv import load_dotenv

###
import pathlib
sys.path.append(str(pathlib.Path().resolve()))
###

load_dotenv() 

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

while True:
  if current_file_path.split("/")[-2] == "Apps":
    break
  
  # print(parent_directory)
  sys.path.append(parent_directory)

  parent_folder = parent_directory.split("/")[-1]
  if parent_folder == "Apps":
    break

  parent_directory = os.path.dirname(parent_directory)

# pprint(sys.path)

cwd = os.getcwd()

cwd_parent = os.path.dirname(cwd)

if current_file_path.split("/")[-2] != "Apps":
  while cwd_parent.split("/")[-1] != "Apps":
    cwd_parent = os.path.dirname(cwd_parent)
  APP_PATH = cwd_parent
else:
  APP_PATH = "/".join(current_file_path.split("/")[:-1])