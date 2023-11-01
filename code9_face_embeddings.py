from deepface import DeepFace
import cv2
import numpy as np
import os
from natsort import natsorted


path = "/home/alp/PycharmProjects/FaceRecognition/real-life-db/images_cleaned_larger(resized)"
images = natsorted([os.path.join(path, file) for file in os.listdir(path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))])
embeddings_path = "/home/alp/PycharmProjects/FaceRecognition/real-life-db/embeddings"
if not os.path.exists(embeddings_path):
    os.makedirs(embeddings_path)

# Yüzleri gömme vektörlerine dönüştürmek için kullanılacak işlev
def get_face_embedding(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV görüntüyü BGR'den RGB'ye dönüştürür
    img_embedding = DeepFace.represent(img_path=img, model_name="VGG-Face", enforce_detection=False)
    return img_embedding

for img in images:
    # Yeni bir yüz fotoğrafını temsil eden gömme vektörünü alın
    new_face_embedding = get_face_embedding(img)

    file_name, _ = os.path.splitext(os.path.basename(img))
    txt_file_name = f"{embeddings_path}/{file_name}.txt"

    with open(txt_file_name, "w") as file:
        for item in new_face_embedding:
            file.write("%s\n" % item)


## Saving face embeddings
# with open("embedding_vector.txt", "w") as file:
#     for item in new_face_embedding:
#         file.write("%s\n" % item)

## Reading face embeddings
# with open("embedding_vector.txt", "r") as file:
#     loaded_embedding = [float(line.strip()) for line in file]





# # Yeni yüzü veri kümesindeki yüzlerle karşılaştırın
# for person, embedding in dataset.items():
#     # Cosine benzerliğini hesapla (1 - cosine_distance)
#     similarity = 1 - np.dot(embedding, new_face_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(new_face_embedding))
#     threshold = 0.3  # Benzerlik eşiği
#     if similarity > threshold:
#         print(f"Benzer kişi: {person}, Benzerlik Skoru: {similarity}")
#     else:
#         print(f"Farklı kişi: {person}, Benzerlik Skoru: {similarity}")
