import cv2

for i in range(10):
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
    if cap.isOpened():
        print("Camara encontrada en indice:", i)
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Camara", frame)
            cv2.waitKey(2000)
        cap.release()

cv2.destroyAllWindows()