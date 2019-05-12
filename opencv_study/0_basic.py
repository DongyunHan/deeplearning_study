import cv2
import math
# flags
# IMREAD_COLOR, => 투명도 정보 X
# IMREAD_GRAYSCALE,
# IMREAD_UNCHANGED => 투명도 정보 O
img_color = cv2.imread('./images/avengers-imax.jpg', cv2.IMREAD_COLOR)
img_gray = cv2.imread('./images/avengers-imax.jpg', cv2.IMREAD_GRAYSCALE)

width = math.floor(1723 * 0.3)
height = math.floor(2500*0.3)

resized_image = cv2.resize(img_color, (width, height))

cv2.namedWindow("Avengers! Assemble")
cv2.imshow("Show Image", resized_image)

cv2.waitKey(0)

img_gray2 = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# 아래 "Show Image"이란 타이틀을 위와 다른 이름으로 호출하면
# 위의 윈도우와 별도의 윈도우가 생성된다
cv2.imshow("Show Image", img_gray2)
# cv2.imshow("Show Image2", img_gray2)

cv2.waitKey(0)

# save image
cv2.imwrite("./images/avengers-gray.jpg", img_gray2)

cv2.destroyAllWindows()
