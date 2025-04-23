# face_detection_example
Assist my friend's assignment and make a example for easy training and inference.

I use yolov8 as my face detection model and trained from the lisa photos in my train folder
Then use deepface to classify henry and lisa.

Since the deepface rely on old version tf, so we need to build up an old python env such as:
```
conda create -n deepface-env python=3.8 -y
conda activate deepface-env
pip install tensorflow==2.12.0
pip install deepface opencv-python-headless
```

And in the end we would run in local env:

```
python inference.py
```
<img width="863" alt="Screenshot 2025-04-23 at 3 27 56 PM" src="https://github.com/user-attachments/assets/726206a5-ec02-40de-bfed-bafafe7af8ec" />
