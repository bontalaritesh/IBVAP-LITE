import os
import urllib.request
import cv2
import numpy as np

os.makedirs("watchlist_images/Suspect_Alpha", exist_ok=True)
os.makedirs("watchlist_images/Agent_Bravo", exist_ok=True)

# Download sample real face portraits from public standard datasets / test images
# or save clear sample face photos
urls = [
    ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg", "watchlist_images/Suspect_Alpha/ref1.jpg"),
    ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg", "watchlist_images/Suspect_Alpha/ref2.jpg"),
    ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/messi5.jpg", "watchlist_images/Agent_Bravo/ref1.jpg"),
]

for url, target in urls:
    try:
        print(f"Downloading sample reference face to {target}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp, open(target, 'wb') as f:
            f.write(resp.read())
        print(f"Saved {target}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

from faces import FaceRecognitionSystem
fr = FaceRecognitionSystem()
ok, msg = fr.enroll_watchlist("watchlist_images")
print(f"Enrollment test result: {ok} | {msg}")
print(f"Enrolled persons: {list(fr.enrolled_faces.keys())}")