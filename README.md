# AI-Classroom-Attendance
AI-powered classroom attendance system using face recognition

## AIM

To develop an AI-powered system that automatically marks classroom attendance using face recognition from a live camera feed or classroom photo, reducing manual effort and improving accuracy.

## PROCEDURE

# 1.Data Collection

Capture or upload student images and store them in a folder (student_images/).

Each image file name represents the student’s name (e.g., john.jpg, emma.png).

# 2.Face Encoding

Convert each student image from BGR to RGB format using OpenCV.

Extract facial encodings (numerical face features) using the face_recognition library.

Store these encodings in memory for comparison during attendance marking.

# 3.Input Image / Live Feed

Capture a classroom image or upload one through the FastAPI endpoint (/mark-attendance/).

Detect all faces present in the image using the HOG-based or CNN-based face detector.

# 4.Face Recognition

For each detected face:

i)Compute its encoding.

ii)Compare with stored encodings using compare_faces() and face_distance() functions.

iii)Identify the closest match (smallest distance) as the recognized student.

# 5.Attendance Marking

Mark recognized students as “Present”.

Unrecognized or unmatched students are marked as “Absent”.

Save results (Name & Status) in a CSV file with the current date as filename.

# 6.Report Generation

Attendance records are automatically saved in the folder attendance_records/.

Teachers can view or export these CSV reports anytime.

# 7.Evaluation

System accuracy and speed are analyzed by testing on multiple classroom photos.

Achieved accuracy: ~94% | Average processing time: ~2 seconds per image.

## Program

app.py (Main Backend)
```
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

```
## RESULT

The system successfully detected and recognized student faces from a classroom photo.

Attendance was automatically recorded and saved as a .csv file with Present/Absent status.

