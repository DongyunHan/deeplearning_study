import cv2


def nothing(x):
    cur_value = cv2.getTrackbarPos("threshold_bar", "Binary")
    print(cur_value)


cv2.namedWindow("Binary")
cv2.createTrackbar("threshold_bar", "Binary", 0, 255, nothing)
cv2.setTrackbarPos("threshold_bar", "Binary", 127)

img_color = cv2.imread("./images/red_ball.png", cv2.IMREAD_COLOR)
img_color = cv2.resize(img_color, (640, 480))
# cv2.imshow("color", img_color)
# cv2.waitKey(0)

img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
# cv2.imshow("gray", img_gray)
# cv2.waitKey(0)

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
        img_gray, threshold_val, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("Binary", img_binary)

    img_result = cv2.bitwise_and(img_color, img_color, mask=img_binary)
    cv2.imshow("Result", img_result)


cv2.destroyAllWindows()
