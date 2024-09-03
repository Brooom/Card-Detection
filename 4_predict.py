import cv2
from ultralytics import YOLO
import os

def getLastModelPath():
    # Get the path of the trained model
    path = f"{os.getcwd()}/runs/detect/"
    folders = os.listdir(path)
    train_folders = [folder for folder in folders if "train" in folder]
    last_train_folder = sorted(train_folders)[-2]
    model_path = f"{path}{last_train_folder}/weights/"
    print(f"Last train folder: {model_path}")
    return model_path


# Load the model
model_path = getLastModelPath()
print(f"Model path: {model_path}")
model = YOLO(f"{model_path}best.pt")

# Predict on video
model.predict("test_video.mp4", save=True)

exit()

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
