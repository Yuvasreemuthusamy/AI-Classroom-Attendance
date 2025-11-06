from fastapi import FastAPI, UploadFile, HTTPException
import face_recognition
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime

app = FastAPI(title="AI-Based Classroom Attendance System")

# --------------------------
# STEP 1: Load Student Images
# --------------------------
path = "student_images"
if not os.path.exists(path):
    os.makedirs(path)

images = []
student_names = []

for file in os.listdir(path):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = cv2.imread(f"{path}/{file}")
        if img is not None:
            images.append(img)
            student_names.append(os.path.splitext(file)[0])

if len(images) == 0:
    print("No student images found in 'student_images/' folder.")

# --------------------------
# STEP 2: Encode Faces
# --------------------------
def find_encodings(images):
    encode_list = []
    for img in images:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)
        if encodings:
            encode_list.append(encodings[0])
    return encode_list

encode_list_known = find_encodings(images)

# --------------------------
# STEP 3: Attendance Endpoint
# --------------------------
@app.post("/mark-attendance/")
async def mark_attendance(file: UploadFile):
    try:
        # Read uploaded image
        img = face_recognition.load_image_file(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Detect and encode faces in uploaded image
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)

    if len(face_encodings) == 0:
        return {"message": "No faces detected in the image."}

    names_present = []

    # Compare with known faces
    for encode_face in face_encodings:
        matches = face_recognition.compare_faces(encode_list_known, encode_face)
        face_dist = face_recognition.face_distance(encode_list_known, encode_face)

        if len(face_dist) > 0:
            match_index = np.argmin(face_dist)
            if matches[match_index]:
                name = student_names[match_index].upper()
                names_present.append(name)

    # Prepare attendance CSV
    date = datetime.now().strftime("%Y-%m-%d")
    os.makedirs("attendance_records", exist_ok=True)

    df = pd.DataFrame({"Name": student_names})
    df["Status"] = ["Present" if name in names_present else "Absent" for name in student_names]
    csv_path = f"attendance_records/attendance_{date}.csv"
    df.to_csv(csv_path, index=False)

    return {
        "message": "Attendance marked successfully.",
        "date": date,
        "present_students": names_present,
        "csv_file": csv_path
    }
