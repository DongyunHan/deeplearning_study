# 영상 처리 알고리즘을 적용하기 전에
# 전처리 단계로 이진화가 사용된다.
# ex) 배경과 오브젝트를 불리하는데 사용

import cv2

# dummy funciton
# necessary to use trackbar


def nothing(x):
    cur_value = cv2.getTrackbarPos("threshold_bar", "Binary")
    print(cur_value)


# 평소에는 있으나 마나한 namedWindow였지만
# trackbar를 붙일 윈도우를 생성해주어야 한다.
cv2.namedWindow("Binary")
# createTrackbar(
#  id => trackbar identifier
#  window name => in this case Binary Trackbar
#  min value
#  max value
#  call-back function
# )
cv2.createTrackbar("threshold_bar", "Binary", 0, 255, nothing)
cv2.setTrackbarPos("threshold_bar", "Binary", 127)

img_color = cv2.imread("./images/red_ball.png", cv2.IMREAD_COLOR)
img_color = cv2.resize(img_color, (640, 480))
cv2.imshow("color", img_color)
cv2.waitKey(0)

img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", img_gray)
cv2.waitKey(0)

# arguments(
# source image => it must be gray scale image
# threshold
# target value => based on Option,
#                 when a pixel value is bigger than threshold
#                 that pixel become to target value.
# OPTION
# )
ARROW_COMMAND = [2, 3]
key_input = ''

while True:

    key_input = cv2.waitKey(1)
    if key_input & 0xFF == 27:
        break

    # make left rifht arrow control the threshold value
    if key_input & 0xFF == 2:
        threshold_val = threshold_val - 1
        print(threshold_val)
        cv2.setTrackbarPos("threshold_bar", "Binary", threshold_val)
    elif key_input & 0xFF == 3:
        threshold_val = threshold_val + 1
        print(threshold_val)
        cv2.setTrackbarPos("threshold_bar", "Binary", threshold_val)
    else:
        threshold_val = cv2.getTrackbarPos("threshold_bar", "Binary")

    ret, img_binary = cv2.threshold(
        img_gray, threshold_val, 255, cv2.THRESH_BINARY)
    cv2.imshow("Binary", img_binary)

    img_result = cv2.bitwise_and(img_color, img_color, mask=img_binary)
    cv2.imshow("Result", img_result)


cv2.destroyAllWindows()
