# Naruto Hand Seals Detection


<img src="https://github.com/Jannat-Javed/Naruto-Hand-Seals-Detection/blob/main/pics/team%207.jpg" alt="team 7" width="600"/>

When I was a kid, I used to watch an anime called Naruto, which followed the adventures of its main protagonist, a young ninja named Naruto. In the world of Naruto, ninjas possessed "chakra," an energy that allowed them to perform powerful jutsus or techniques. To activate these techniques, they had to perform specific hand seals or hand signs, which helped channel their chakra. As a child, I would excitedly mimic these hand signs and shout out the names of different techniques.
This project utilizes **YOLOv11** to perform real-time detection of Naruto hand seals. The system processes input from a webcam or video file, identifying 12 different Naruto hand seals, and displaying the detected seals along with their corresponding names and confidence scores in the video stream. It also overlays custom images for each hand seal.
There are 12 basic hand signs or hand seals, each one named after an animal in the Chinese Zodiac.

<p align="center">
  <div style="display: flex; justify-content: center;">
    <img src="https://github.com/Jannat-Javed/Naruto-Hand-Seals-Detection/blob/main/pics/Kakashi.gif" alt="GIF 1" width="400" style="margin-right: 20px;"/>
    <img src="https://github.com/Jannat-Javed/Naruto-Hand-Seals-Detection/blob/main/pics/Minato.gif" alt="GIF 2" width="400"/>
  </div>
</p>






## Features

- **Real-Time Detection** of Naruto hand seals from webcam or video.
- **Smooth Performance** with FPS calculation displayed on screen.
- **Customizable Display** for each detected hand seal with its image.
- **Confidence Threshold** to adjust detection accuracy.
- **Adjustable Frame Processing** for optimized performance.

## Dataset
This project is designed for detecting Naruto hand seals. I used the [Naruto Hand Sign](https://www.kaggle.com/datasets/vikranthkanumuru/naruto-hand-sign-dataset) dataset.
The model was trained on a dataset that includes the following hand seals:

- **Bird**
- **Boar**
- **Dog**
- **Dragon**
- **Hare (Rabbit)**
- **Horse**
- **Monkey**
- **Ox**
- **Ram (Sheep)**
- **Rat**
- **Snake**
- **Tiger**

<img src="https://github.com/Jannat-Javed/Naruto-Hand-Seals-Detection/blob/main/pics/hand%20seals.png" alt="hand seals" width="600"/>


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
- main.py: Contains the logic for processing detections and displaying pics.



