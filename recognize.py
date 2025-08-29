import cv2
import os
import csv
from datetime import datetime

# Face detector
proto = "deploy.prototxt"
model = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(proto, model)

# Paths
model_path = "face_model.xml"
attendance_file = "attendance.csv"

# Load trained LBPH model
recognizer = cv2.face.LBPHFaceRecognizer_create()
if os.path.exists(model_path):
    recognizer.read(model_path)
else:
    print("No trained model found. Run train.py first.")
    exit()

# Ensure attendance.csv exists with headers
if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

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

def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")

    # Read existing attendance to prevent duplicate entries per day
    already_marked = False
    if os.path.exists(attendance_file):
        with open(attendance_file, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                if row and row[0] == name and row[1] == today:
                    already_marked = True
                    break

    if not already_marked:
        with open(attendance_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, today, time_now])
        print(f"Attendance marked for {name} at {time_now}")

def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = detect_face(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for (x1, y1, x2, y2) in faces:
            face_gray = gray[y1:y2, x1:x2]
            if face_gray.size == 0:
                continue
            label, confidence = recognizer.predict(face_gray)
            
            # Fetch the stored name
            name = recognizer.getLabelInfo(label)
            if not name:
                name = "Unknown"
            
            # Mark attendance if recognized and confidence is good
            if name != "Unknown" and confidence < 80:  # threshold to reduce false positives
                mark_attendance(name)

            text = f"{name} ({confidence:.2f})"
            cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        cv2.imshow("Recognition & Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
