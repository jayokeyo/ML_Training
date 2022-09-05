import os
from langdetect import detect

file = open(os.path(input('Path to file: ')))
print(detect(file))