import langdetect
import os

file_path = os.path.abspath(input('Path to file: '))
file = str(open(file_path))
print(langdetect.detect(file))
