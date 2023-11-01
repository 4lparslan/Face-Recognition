import os
from natsort import natsorted
import cv2
from deepface import DeepFace
import numpy as np
import time

start_time = time.time()

def get_face_embedding(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV görüntüyü BGR'den RGB'ye dönüştürür
    img_embedding = DeepFace.represent(img_path=img, model_name="VGG-Face", enforce_detection=False)
    return img_embedding

embeddings_path = "/home/alp/PycharmProjects/FaceRecognition/real-life-db/embeddings"
txt_files = natsorted([os.path.join(embeddings_path, file) for file in os.listdir(embeddings_path) if file.lower().endswith('.txt')])
txt_list = {}
for txt in txt_files:
    with open(txt, "r") as file:
        loaded_embedding = [line.strip() for line in file]
        basename = os.path.basename(txt)
        txt_list[basename] = loaded_embedding
        # txt_list = {"1.txt" = [....], "2.txt" = [....],}

print("Embeddings Loaded")

new_face_embedding = get_face_embedding("/home/alp/PycharmProjects/FaceRecognition/real-life-db/base.png")

for filenm, embedding in txt_list.items():
    for embed in embedding:
        embed = eval(embed)

        dissimilarity = 1 - np.dot(embed['embedding'], new_face_embedding[0]['embedding']) / (
                np.linalg.norm(embed['embedding']) * np.linalg.norm(new_face_embedding[0]['embedding']))

        threshold = 0.3  # Benzerlik eşiği
        if dissimilarity < threshold:
            print(f"Benzer kişi: {filenm}, Benzerlik Skoru: {dissimilarity}")

            break
        else:
            #print(f"Farklı kişi: {filenm}, Benzerlik Skoru: {dissimilarity}")
            pass

end_time = time.time()

passing_time = end_time - start_time
print(passing_time)