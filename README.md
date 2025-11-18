# Design and Development of a Real-Time Gesture-Controlled Fruit-Slicing Game Using Computer Vision
This project uses modern computer-vision techniques to create an interactive, gesture-controlled fruit-slicing game. Using a standard webcam together with OpenCV, MediaPipe Hands, and NumPy, the system tracks the player’s index finger in real time and turns it into a “blade” that slices falling fruit icons on the screen. A simple game loop manages fruit spawning, collision detection, scoring, lives, and dynamic difficulty to deliver a responsive and engaging experience.
![](images/gameplay_example.png)

## Overview and Background
Controlling a game directly with hand gestures requires robust, low-latency tracking of the player’s hand and precise mapping from real-world motion to on-screen interactions. In this project, the webcam feed is processed frame-by-frame, and the MediaPipe Hands model is used to detect hand landmarks and extract the position of the index fingertip. This fingertip position is visualized as a colored dot and a trailing “slash” line, which acts as the cutting blade.

Colorful fruit sprites (apple, banana, orange, strawberry, etc.) are rendered using OpenCV and move across the screen with configurable velocities. When the distance between the fingertip and a fruit falls below a radius threshold, the fruit is considered “sliced”, the score is increased, and the slash color momentarily changes to match the fruit. Missed fruit reduce the player’s lives, and when lives reach zero the game transitions to a game-over state.

The system includes a lightweight start menu where the camera feed is shown together with on-screen instructions; pressing **Enter** starts the game loop. As the score grows, the difficulty automatically scales by increasing fruit speed and spawn rate, creating a progressively more challenging experience. Because it relies only on a webcam and common Python libraries (OpenCV, MediaPipe, NumPy), the project is easy to run, experiment with, and extend—for example by adding new fruit types, combo systems, or additional gesture-based mechanics.

## Table of Contents
```
ninja-fruit-game
|__ images
|   |__ apple.png
|   |__ banana.png
|   |__ orange.png
|   |__ strawberry.png
|__ src
    |__ main.py
README.md
requirements.txt
LICENSE
```

## Getting started

