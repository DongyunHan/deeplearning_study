import cv2

cap = cv2.VideoCapture('./videos/output.avi')
# fourcc = cv2.VideoWriter_fourcc(*"XVID")

while True:
    ret, img_color = cap.read()

    if ret == False:
        continue

    cv2.imshow("showing", img_color)
    print("is it wokring???")

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
