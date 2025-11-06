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

## RESULT

The system successfully detected and recognized student faces from a classroom photo.

Attendance was automatically recorded and saved as a .csv file with Present/Absent status.

