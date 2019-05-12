import cv2

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*"XVID")
# file name,
# codec
# frame
# video size => 캡쳐되는 이미지 사이즈와 같아야함
writer = cv2.VideoWriter("./videos/output.avi", fourcc, 30.0, (640, 480))

while True:
    ret, img_color = cap.read()

    if ret == False:
        continue

    img_color = cv2.resize(img_color, (640, 480))

    cv2.imshow("recording...", img_color)
    writer.write(img_color)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
