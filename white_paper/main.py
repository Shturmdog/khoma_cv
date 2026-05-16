from pickletools import uint8

import cv2
import numpy as np
import zmq

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.setsockopt(zmq.SUBSCRIBE, b"")
socket.connect("tcp://84.237.21.36:6002")

cv2.namedWindow("Stream", cv2.WINDOW_GUI_NORMAL)
count = 0


def draw_rotated_text(img, text, center, angle, font_scale=0.7, color=(255, 0, 0), thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)

    text_img = np.zeros((h + 20, w + 20, 4), dtype=np.uint8)
    cv2.putText(text_img, text, (10, h + 10), font, font_scale, (*color, 255), thickness)


    M = cv2.getRotationMatrix2D((text_img.shape[1] / 2, text_img.shape[0] / 2), angle, 1)
    rotated = cv2.warpAffine(text_img, M, (text_img.shape[1], text_img.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    x, y = int(center[0] - rotated.shape[1] / 2), int(center[1] - rotated.shape[0] / 2)

    x_end = min(img.shape[1], x + rotated.shape[1])
    y_end = min(img.shape[0], y + rotated.shape[0])
    rotated = rotated[0:y_end - y, 0:x_end - x]


    alpha = rotated[:, :, 3] / 255.0
    for c in range(3):
        img[y:y + rotated.shape[0], x:x + rotated.shape[1], c] = \
            (1 - alpha) * img[y:y + rotated.shape[0], x:x + rotated.shape[1], c] + alpha * rotated[:, :, c]


while True:
    msg = socket.recv()
    key = cv2.waitKey(100)
    if key == ord('q'):
        break
    count += 1
    frame = cv2.imdecode(np.frombuffer(msg, np.uint8), -1)
    if frame is None:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)

        M = cv2.moments(max_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

        rect = cv2.minAreaRect(max_contour)
        angle = rect[2]
        if angle < -45:
            angle += 90


        draw_rotated_text(frame, "BoB", (cx, cy), angle, font_scale=0.7, color=(255, 0, 0), thickness=2)

    cv2.putText(frame, f"Count frame {count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0))
    cv2.imshow("Stream", frame)

