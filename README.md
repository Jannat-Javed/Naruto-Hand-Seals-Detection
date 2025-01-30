# Naruto Hand Seals Detection


![undefined - Imgur (2)](https://github.com/user-attachments/assets/60f1f31c-55ba-42a0-89f2-1e2a4422fd65)

![undefined - Imgur (3)](https://github.com/user-attachments/assets/b4bbc773-2173-4230-a30f-00df73fa3da4)




When I was a kid, I used to watch an anime called Naruto, which followed the adventures of its main protagonist, a young ninja named Naruto. In the world of Naruto, ninjas possessed "chakra," an energy that allowed them to perform powerful jutsus or techniques. To activate these techniques, they had to perform specific hand seals or hand signs, which helped channel their chakra. As a child, I would excitedly mimic these hand signs and shout out the names of different techniques—though, in reality, I had no idea what the actual hand seals were. I just made them up as I went along.

This project utilizes **YOLOv11** to perform real-time detection of Naruto hand seals. The system processes input from a webcam or video file, identifying 12 different Naruto hand seals, and displaying the detected seals along with their corresponding names and confidence scores in the video stream. It also overlays custom images for each hand seal.
There are 12 basic hand signs or hand seals, each one named after an animal in the Chinese Zodiac.

## Features

- **Real-Time Detection** of Naruto hand seals from webcam or video.
- **Smooth Performance** with FPS calculation displayed on screen.
- **Customizable Display** for each detected hand seal with its image.
- **Confidence Threshold** to adjust detection accuracy.
- **Adjustable Frame Processing** for optimized performance.
- **User-Friendly Command-Line Interface** with customizable settings.

## Dataset
This project is designed for detecting Naruto hand seals. I used the [Naruto Hand Sign](https://www.kaggle.com/datasets/vikranthkanumuru/naruto-hand-sign-dataset) dataset.
The model was trained on a dataset that includes the following hand seals:

- **Bird**
- **Boar**
- **Dog**
- **Dragon**
- **Hare**
- **Horse**
- **Monkey**
- **Ox**
- **Ram**
- **Rat**
- **Snake**
- **Tiger**

## Requirements

Before running the project, ensure you have the following:

- Python 
- OpenCV
- Ultralytics 
- Argparse
- Pytorch
- Roboflow


## Files
 - demo.py: Main script for real-time detection.
- main.py: Contains the logic for processing detections and displaying results.
- train.ipynb: Notebook for training the YOLOv11 model.



