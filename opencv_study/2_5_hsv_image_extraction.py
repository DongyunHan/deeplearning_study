import cv2
import numpy as np

hsv = 0
lower_blue1 = 0
upper_blue1 = 0
lower_blue2 = 0
upper_blue2 = 0
lower_blue3 = 0
upper_blue3 = 0


def clickHandler(event, x, y, flags, param):
    global hsv, lower_blue1, upper_blue1, lower_blue2, upper_blue2, lower_blue3, upper_blue3

    if event == cv2.EVENT_LBUTTONDOWN:
        # print(y, x)
        # print(img_color[y, x])
        color = img_color[y, x]
        pixel_val = np.uint8([[color]])

        hsv = cv2.cvtColor(pixel_val, cv2.COLOR_BGR2HSV)
        hsv = hsv[0][0]

        if hsv[0] < 10:
            print("case1")
            lower_blue1 = np.array([hsv[0]-10+180, 30, 30])
            upper_blue1 = np.array([180, 255, 255])
            lower_blue2 = np.array([0, 30, 30])
            upper_blue2 = np.array([hsv[0], 255, 255])
            lower_blue3 = np.array([hsv[0], 30, 30])
            upper_blue3 = np.array([hsv[0]+10, 255, 255])

        elif hsv[0] > 170:
            print("case2")
            lower_blue1 = np.array([hsv[0], 30, 30])
            upper_blue1 = np.array([180, 255, 255])
            lower_blue2 = np.array([0, 30, 30])
            upper_blue2 = np.array([hsv[0]+10-180, 255, 255])
            lower_blue3 = np.array([hsv[0]-10, 30, 30])
            upper_blue3 = np.array([hsv[0], 255, 255])
        else:
            print("case3")
            lower_blue1 = np.array([hsv[0], 30, 30])
            upper_blue1 = np.array([hsv[0]+10, 255, 255])
            lower_blue2 = np.array([hsv[0]-10, 30, 30])
            upper_blue2 = np.array([hsv[0], 255, 255])
            lower_blue3 = np.array([hsv[0]-10, 30, 30])
            upper_blue3 = np.array([hsv[0], 255, 255])

        print(hsv[0])
        print("@1", lower_blue1, "~", upper_blue1)
        print("@2", lower_blue2, "~", upper_blue2)
        print("@3", lower_blue3, "~", upper_blue3)


cv2.namedWindow("hsv_circle")
cv2.setMouseCallback("hsv_circle", clickHandler)

img_color = cv2.imread("./images/hsv_circle.jpg")
height, width = img_color.shape[:2]
img_color = cv2.resize(img_color, (width, height),
                       interpolation=cv2.INTER_AREA)
img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

while True:
    # 범위 값으로 HSV 이미지에서 마스크를 생성합니다.
    img_mask1 = cv2.inRange(img_hsv, lower_blue1, upper_blue1)
    img_mask2 = cv2.inRange(img_hsv, lower_blue2, upper_blue2)
    img_mask3 = cv2.inRange(img_hsv, lower_blue3, upper_blue3)
    img_mask = img_mask1 | img_mask2 | img_mask3

    img_result = cv2.bitwise_and(img_color, img_color, mask=img_mask)

    cv2.imshow("hsv_circle", img_color)
    cv2.imshow("img_mask1", img_mask1)
    cv2.imshow("img_mask2", img_mask2)
    cv2.imshow("img_mask3", img_mask3)
    cv2.imshow("img_mask", img_mask)

    cv2.imshow("img_result", img_result)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
