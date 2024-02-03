import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def read_text_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read().splitlines()
    return np.array(data, dtype=float)

file2_data = read_text_file('flateen.txt')
file1_data = read_text_file('crop.txt')
cosine_value = cosine_similarity([file1_data], [file2_data])[0][0]
print(cosine_value)