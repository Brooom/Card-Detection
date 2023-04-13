
#%%
import cv2
from ultralytics import YOLO








model = YOLO("runs\\detect\\train37\\weights\\best.pt")

#model.predict(source="C:\\Users\\beatb\\OneDrive\\Documents\\Programmieren\\Card-Detection\\test_video.mp4", save=True)

"""
# Open the video file
video_path = "C:\\Users\\beatb\\OneDrive\\Documents\\Programmieren\\Card-Detection\\test_video.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model.predict(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    else:
        # Break the loop if the end of the video is reached
        break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
    """

#Predict cam

cap = cv2.VideoCapture(0)
while True:
    # Read a frame from the video
    success, frame = cap.read()


    results = model.predict(frame, conf=0.5)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    cv2.waitKey(10)



    # Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
