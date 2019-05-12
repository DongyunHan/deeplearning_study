import cv2


# 비디오 두 개 활용시
# argument값이 0, 1 인 비디오 캡처 객체를 두 개 생성하면 된다.
cap = cv2.VideoCapture(0)

while True:
    ret, img_color = cap.read()

    # 가끔씩 캡처가 실패할 수도 있기 때문에 실패 처리
    if ret == False:
        continue

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    cv2.imshow("cap", img_color)
    cv2.imshow("gray cap", img_gray)

    # 키보드 입력을 받기 위해서 대기시간 1초를 준다.
    # typed key & 0xFF (00000000000000000000000011111111) == ESC
    # 이면 break

    # cv2.waitKey( n )
    # n밀리초 후에 다음 프레임으로 넘어감 (n 밀리초 동안 키보드 입력을 기다린 후 다음 코드가 진행)

    # The function waitKey() waits for key event for a "delay".
    # As explained in the OpenCV documentation, HighGui (imshow() is a function of HighGui)
    # need a call of waitKey regularly, in order to process its event loop.

    # if you don't call waitKey, HighGui cannot process windows events like
    # redraw, resizing, input event etc.
    # So just call it, even with a 1ms delay.
    # -------from stackoverflow---------
    if cv2.waitKey(2) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
