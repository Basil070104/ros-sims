import struct
import torch
import os
import pickle
import mimetypes

PATH = "maps/386-40-False.map"
# if os.path.exists(PATH):
#   map_data = torch.load(f=PATH, map_location="cpu", pickle_module=pickle)
  
mime_type = mimetypes.guess_type(PATH)[0]
print(mime_type)

if mime_type:
    print(f"The MIME type of the file is: {mime_type}")
else:
    print("Could not determine the file type.")
    
try:
  state = torch.load("pth/130.pth")
  map = torch.load("maps/386-40-False.map")
  print(state)
except FileNotFoundError:
  print("File not found")

  
# print(map_data)
