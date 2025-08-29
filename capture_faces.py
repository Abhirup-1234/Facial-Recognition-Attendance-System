import cv2
import os

# Paths
dataset_path = "dataset"
os.makedirs(dataset_path, exist_ok=True)

# Face detector (DNN)
proto = "deploy.prototxt"
model = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(proto, model)

def detect_face(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")
            faces.append((x1, y1, x2, y2))
    return faces

def main():
    # Ask only for name
    user_name = input("Enter Name: ").strip()
    
    cap = cv2.VideoCapture(0)
    count = 0
    capturing = False

    print("Press 'c' to start capturing faces, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_face(frame)
        for (x1, y1, x2, y2) in faces:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Save only if capturing is enabled
            if capturing and count < 30:
                face = frame[y1:y2, x1:x2]
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                filename = f"{dataset_path}/{user_name}_{count}.jpg"
                cv2.imwrite(filename, face_gray)
                count += 1

        # Show status on screen
        if capturing:
            cv2.putText(frame, f"Capturing... {count}/30",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Capture Faces", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            capturing = True
            print("Started capturing faces...")
        elif key == ord('q') or count >= 30:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Face capture complete. Saved {count} images for {user_name}.")

if __name__ == "__main__":
    main()
