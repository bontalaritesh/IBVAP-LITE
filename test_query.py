import cv2
from faces import FaceRecognitionSystem

fr = FaceRecognitionSystem()
img = cv2.imread("watchlist_images/Suspect_Alpha/ref1.jpg")
results = fr.process_frame(img, threshold=0.40)
print("=== Face Recognition Match Results ===")
for r in results:
    print(f"Name: {r['name']} | Confidence: {r['confidence']*100:.1f}% | IsMatch: {r['is_match']} | Bbox: {r['bbox']}")