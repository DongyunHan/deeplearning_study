# 시스템에 따라서 이미지의 픽셀 위치를 표현하는 값이 다를 수 있다.
# (y,x) or (x,y)

# 시스템에 따라서 색상을 표현하는 비트수가 다르고 그에 따른 색상차이가 생길 수 있다.

# 시스템에 따라서 이미지를 표현하는 색상 표현법이 다를 수 있다.
# 기본적으로 RGB 채널을 활용하지만 BGR 채널이나 채널이 바뀌어 있는 경우
# 그려진 이미지는 다르게 표현된다. (간단하게 채널을 바꿔주면 해결)

import numpy as np
import cv2

color = [255, 0, 0]
# opencv 함수가 인식할 수 있도록 color값을
# 한 픽셀로 구성된 이미지로 변환
pixel = np.uint8([[color]])

#################
# HSV
# H: Hue -> 색상
# S: Saturation -> 색의 강도(채도)
# V: Value ->  빛의 강도(명도)
#################
hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
hsv = hsv[0][0]

print("bgr: ", color)
print("hsv: ", hsv)
