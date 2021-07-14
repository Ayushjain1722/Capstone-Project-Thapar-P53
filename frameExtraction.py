# Program To Read video
# and Extract Frames
import cv2

vidObj = cv2.VideoCapture("C:\\Users\\ayush\\Desktop\\7th Semester\\4. Capstone Project\\Test Videos\\videoplayback.mp4")
count = 0
# checks whether frames were extracted
success = 1

while success:
    # vidObj object calls read
    # function extract frames
    success, image = vidObj.read()

    # Saves the frames with frame-count
    cv2.imwrite("C\\Users\\ayush\\frames\\frame%d.jpg" % count, image)

    count += 1