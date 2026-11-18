import cv2

print("Testing camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera!")
    print("Try changing index from 0 to 1, 2, etc.")
    exit(1)

print("Camera opened! Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not read frame!")
        break
    
    cv2.imshow('Camera Test', frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Camera test complete!")
