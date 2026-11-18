import sys
import cv2

video, frame_no = sys.argv[1], int(sys.argv[2])
cap = cv2.VideoCapture(video)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
ok, img = cap.read()
pts = []

def click(event, x, y, *_):
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))
        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
        cv2.putText(img, str(len(pts)), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("pick", img)
        print(len(pts), (x, y))

cv2.imshow("pick", img)
cv2.setMouseCallback("pick", click)
cv2.waitKey(0)                 # press any key to close when done
cv2.destroyAllWindows()
print("points:", pts)