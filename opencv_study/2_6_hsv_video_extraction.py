import cv2
import numpy as np

hsv = 0
lower_blue1 = 0
upper_blue1 = 0
lower_blue2 = 0
upper_blue2 = 0
lower_blue3 = 0
upper_blue3 = 0

WIDTH = 640
HEIGHT = 480

isChanged = False


def nothing(x):
    pass


def clickHandler(event, x, y, flags, param):
    global hsv, lower_blue1, upper_blue1, lower_blue2, upper_blue2, lower_blue3, upper_blue3

    print("clickHandler - callback")

    if event == cv2.EVENT_LBUTTONDOWN:
        # isChanged = True
        # print(y, x)
        # print(img_color[y, x])
        color = img_color[y, x]
        pixel_val = np.uint8([[color]])

        sat_val = cv2.getTrackbarPos("threshold", "img_result")

        hsv = cv2.cvtColor(pixel_val, cv2.COLOR_BGR2HSV)
        hsv = hsv[0][0]

        if hsv[0] < 10:
            print("case1")
            lower_blue1 = np.array([hsv[0]-10+180, sat_val, sat_val])
            upper_blue1 = np.array([180, 255, 255])
            lower_blue2 = np.array([0, sat_val, sat_val])
            upper_blue2 = np.array([hsv[0], 255, 255])
            lower_blue3 = np.array([hsv[0], sat_val, sat_val])
            upper_blue3 = np.array([hsv[0]+10, 255, 255])

        elif hsv[0] > 170:
            print("case2")
            lower_blue1 = np.array([hsv[0], sat_val, sat_val])
            upper_blue1 = np.array([180, 255, 255])
            lower_blue2 = np.array([0, sat_val, sat_val])
            upper_blue2 = np.array([hsv[0]+10-180, 255, 255])
            lower_blue3 = np.array([hsv[0]-10, sat_val, sat_val])
            upper_blue3 = np.array([hsv[0], 255, 255])
        else:
            print("case3")
            lower_blue1 = np.array([hsv[0], sat_val, sat_val])
            upper_blue1 = np.array([hsv[0]+10, 255, 255])
            lower_blue2 = np.array([hsv[0]-10, sat_val, sat_val])
            upper_blue2 = np.array([hsv[0], 255, 255])
            lower_blue3 = np.array([hsv[0]-10, sat_val, sat_val])
            upper_blue3 = np.array([hsv[0], 255, 255])

        print(hsv[0])
        print("@1", lower_blue1, "~", upper_blue1)
        print("@2", lower_blue2, "~", upper_blue2)
        print("@3", lower_blue3, "~", upper_blue3)


cv2.namedWindow("hsv_circle")
cv2.setMouseCallback("hsv_circle", clickHandler)

cv2.namedWindow("img_result")
cv2.createTrackbar("threshold", "img_result", 0, 255, nothing)
cv2.setTrackbarPos("threshold", "img_result", 30)

cap = cv2.VideoCapture(0)
# cap.set(3, WIDTH)
# cap.set(4, HEIGHT)
cap.set(5, 20)

while True:
    ret, img_color = cap.read()
    img_color = cv2.resize(img_color, (WIDTH, HEIGHT),
                           interpolation=cv2.INTER_AREA)
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

    # 범위 값으로 HSV 이미지에서 마스크를 생성합니다.

    img_mask1 = cv2.inRange(img_hsv, lower_blue1, upper_blue1)
    img_mask2 = cv2.inRange(img_hsv, lower_blue2, upper_blue2)
    img_mask3 = cv2.inRange(img_hsv, lower_blue3, upper_blue3)
    img_mask = img_mask1 | img_mask2 | img_mask3

    # apply morphology
    # I will handle this next Time
    # just know that it will help to remove noise
    kernel = np.ones((11, 11), np.uint8)
    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel)
    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel)

    img_result = cv2.bitwise_and(img_color, img_color, mask=img_mask)

    # tracking targets
    numOfLabels, img_label, stats, centroids = cv2.connectedComponentsWithStats(
        img_mask)
    for idx, centroid in enumerate(centroids):
        if stats[idx][0] == 0 and stats[idx][1] == 0:
            continue

        if np.any(np.isnan(centroid)):
            continue

        x, y, width, height, area = stats[idx]
        centerX, centerY = int(centroid[0]), int(centroid[1])

        if area > 50:
            cv2.circle(img_color, (centerX, centerY), 10, (0, 0, 255), 10)
            cv2.rectangle(img_color, (x, y), (x+width, y+width), (0, 0, 255))

    cv2.imshow("hsv_circle", img_color)
    cv2.imshow("img_mask", img_mask)
    cv2.imshow("img_result", img_result)

    # isChanged = False

    if cv2.waitKey(5) & 0xFF == 27:
        break

cv2.destroyAllWindows()
