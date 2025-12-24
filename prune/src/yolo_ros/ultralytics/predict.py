from ultralytics import YOLO
import os

if __name__ == '__main__':

    model = YOLO(r"best.pt")


    image_folder = r"testimg"


    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_folder, filename)

            results = model(source=image_path, show=True, save=True, classes=[1], conf=0.7)

