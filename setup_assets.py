
import os
import cv2
import numpy as np

# Create watchlist_images directory structure
os.makedirs("watchlist_images/Agent_Alex", exist_ok=True)
os.makedirs("watchlist_images/Suspect_Bravo", exist_ok=True)

# Generate sample portrait images
def create_sample_portrait(filename, name_tag, hue=30):
    img = np.full((320, 320, 3), 40, dtype=np.uint8)
    # Background gradient
    for y in range(320):
        img[y, :] = int(30 + y * 0.15)
    
    # Body silhouette
    cv2.ellipse(img, (160, 310), (120, 90), 0, 0, 360, (70, 70, 80), -1)
    
    # Face structure
    cv2.ellipse(img, (160, 150), (65, 85), 0, 0, 360, (200, 185, 170), -1)
    
    # Hair
    cv2.ellipse(img, (160, 100), (70, 45), 0, 180, 360, (40, 30, 20), -1)
    
    # Eyes
    cv2.circle(img, (135, 140), 7, (255, 255, 255), -1)
    cv2.circle(img, (135, 140), 3, (40, 40, 40), -1)
    cv2.circle(img, (185, 140), 7, (255, 255, 255), -1)
    cv2.circle(img, (185, 140), 3, (40, 40, 40), -1)
    
    # Eyebrows
    cv2.line(img, (120, 125), (150, 128), (40, 30, 20), 3)
    cv2.line(img, (170, 128), (200, 125), (40, 30, 20), 3)
    
    # Nose & Mouth
    cv2.line(img, (160, 145), (160, 170), (160, 140, 120), 2)
    cv2.ellipse(img, (160, 195), (20, 8), 0, 0, 180, (140, 80, 80), -1)
    
    # Text badge
    cv2.putText(img, f"WATCHLIST: {name_tag}", (15, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(filename, img)
    print(f"Created sample portrait: {filename}")

create_sample_portrait("watchlist_images/Agent_Alex/ref_front.jpg", "Alex (01)")
create_sample_portrait("watchlist_images/Agent_Alex/ref_angle.jpg", "Alex (02)")
create_sample_portrait("watchlist_images/Suspect_Bravo/suspect_mugshot.jpg", "Bravo (01)")

from capture import create_synthetic_test_video
create_synthetic_test_video("test_feed.mp4", duration_sec=5, fps=25)
print("Asset setup complete.")
