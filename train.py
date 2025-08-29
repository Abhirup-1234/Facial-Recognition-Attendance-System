import cv2
import os
import numpy as np

dataset_path = "dataset"
model_path = "face_model.xml"

recognizer = cv2.face.LBPHFaceRecognizer_create()

def main():
    faces, labels = [], []
    name_to_label = {}
    current_label = 0

    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg"):
            # Extract the name from filename (before "_")
            name = filename.split("_")[0]

            if name not in name_to_label:
                name_to_label[name] = current_label
                current_label += 1

            img = cv2.imread(os.path.join(dataset_path, filename), cv2.IMREAD_GRAYSCALE)
            faces.append(img)
            labels.append(name_to_label[name])

    # Train LBPH model
    recognizer.train(faces, np.array(labels))

    # Store label->name mapping inside the model itself
    for name, label in name_to_label.items():
        recognizer.setLabelInfo(label, name)

    recognizer.save(model_path)
    print("Training complete. Model + names saved in face_model.xml")

if __name__ == "__main__":
    main()
