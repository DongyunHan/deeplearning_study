import cv2
import math

WIDTH = math.floor(1723 * 0.3)
HEIGHT = math.floor(2500*0.3)

img_color = cv2.imread("./images/avengers-imax.jpg")
img_color = cv2.resize(img_color, (WIDTH, HEIGHT))

img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

lower_blue = (120 - 10, 20, 20)
upper_blue = (120, 255, 255)

# img_hsv의 픽셀값들 중에서
# lower_blue < pixel_value < upper_blue
# 범위 내의 값은 255 범위 밖의 값은 0 인 마스크를 얻는다.

# hsv 값의 hue 값을 활용하여 색을 인식시키는 것은 간단하지만
# 조명에 따라 색이 달라 보일 수 있다든지 (이미지 말고 카메라로 테스트할 시에)
# 두 색의 경계에 해당하는 색을 찾는다든지
# 고려해야하는 가지수가 있음을 인지 할 것.
img_mask = cv2.inRange(img_hsv, lower_blue, upper_blue)

img_result = cv2.bitwise_and(img_color, img_color, mask=img_mask)

cv2.imshow("original", img_color)
cv2.imshow("hsv extracting", img_result)

cv2.waitKey(0)

cv2.destroyAllWindows()
