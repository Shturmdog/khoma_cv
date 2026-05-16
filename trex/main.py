import cv2
import numpy as np
import pyautogui
from mss import mss
import tkinter as tk
from PIL import Image, ImageTk
import time

root = tk.Tk()
root.title("Chrome Dino")
root.attributes('-topmost', True)
label = tk.Label(root)
label.pack()


monitor = {"top": 200, "left": 400, "width": 1000, "height": 300}

#ПАРАМЕТРЫ БЛИЖНЕГО БОКСА
BASE_NEAR_X = 290
NEAR_Y = 150
NEAR_W, NEAR_H = 40, 80

# ПАРАМЕТРЫ ДАЛЬНЕГО БОКСА
BASE_FAR_X = 450
FAR_Y = 150
FAR_W, FAR_H = 60, 80

#ДЕТЕКЦИЯ
BINARY_THRESH = 100
MIN_DARK_RATIO = 0.05

#ПАРАМЕТРЫ СКОРОСТИ
INITIAL_SPEED = 6.0
MAX_SPEED = 13.0
ACCEL = 0.001
FPS = 60
MAX_SHIFT = 100

#ЗАЩИТА ОТ ПОВТОРНЫХ ПРЫЖКОВ
JUMP_COOLDOWN = 0.35
last_jump_time = 0

# Глобальные переменные
sct = mss()
cactus_inside_prev = False
jump_counter = 0
jump_triggered_this_cactus = False
game_start_time = None


def get_current_speed():
    if game_start_time is None:
        return INITIAL_SPEED
    elapsed = time.time() - game_start_time
    speed = INITIAL_SPEED + ACCEL * (elapsed * FPS)
    return min(speed, MAX_SPEED)


def update_box_positions():
    speed = get_current_speed()
    t = (speed - INITIAL_SPEED) / (MAX_SPEED - INITIAL_SPEED)
    shift = int(MAX_SHIFT * t)
    near_x = BASE_NEAR_X - shift
    far_x = BASE_FAR_X - shift
    near_x = max(near_x, 200)
    far_x = max(far_x, near_x + 50)
    return near_x, far_x, shift


def detect_obstacle(roi_gray):
    _, binary = cv2.threshold(roi_gray, BINARY_THRESH, 255, cv2.THRESH_BINARY_INV)
    dark_pixels = cv2.countNonZero(binary)
    total = roi_gray.shape[0] * roi_gray.shape[1]
    return (dark_pixels / total) > MIN_DARK_RATIO

def update_frame():
    global cactus_inside_prev, jump_counter, jump_triggered_this_cactus, last_jump_time, game_start_time

    # Захват
    img = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Позиции боксов
    near_x, far_x, shift = update_box_positions()

    # Ближний бокс
    roi_near = frame[NEAR_Y:NEAR_Y + NEAR_H, near_x:near_x + NEAR_W]
    gray_near = cv2.cvtColor(roi_near, cv2.COLOR_BGR2GRAY)
    obstacle = detect_obstacle(gray_near)

    # Дальний бокс (для определения серий)
    roi_far = frame[FAR_Y:FAR_Y + FAR_H, far_x:far_x + FAR_W]
    gray_far = cv2.cvtColor(roi_far, cv2.COLOR_BGR2GRAY)
    far_obstacle = detect_obstacle(gray_far)

    # Логика прыжка (при входе в ближний бокс)
    if not cactus_inside_prev and obstacle and not jump_triggered_this_cactus:
        now = time.time()
        if now - last_jump_time >= JUMP_COOLDOWN:
            pyautogui.press('space')
            jump_counter += 1

            # Быстрое приземление, если дальний бокс уже показывает препятствие (серия)
            if far_obstacle:
                time.sleep(0.05)
                pyautogui.keyDown('down')
                time.sleep(0.2)
                pyautogui.keyUp('down')

            last_jump_time = now
            jump_triggered_this_cactus = True

            # Фиксируем время старта игры после первого прыжка
            if game_start_time is None:
                game_start_time = time.time()

    # Сброс флага, когда препятствие выходит из ближнего бокса
    if cactus_inside_prev and not obstacle:
        jump_triggered_this_cactus = False

    cactus_inside_prev = obstacle

    # Визуализация
    color_near = (0, 0, 255) if obstacle else (0, 255, 0)
    cv2.rectangle(frame, (near_x, NEAR_Y), (near_x + NEAR_W, NEAR_Y + NEAR_H), color_near, 2)
    color_far = (255, 0, 0) if far_obstacle else (255, 255, 0)
    cv2.rectangle(frame, (far_x, FAR_Y), (far_x + FAR_W, FAR_Y + FAR_H), color_far, 2)

    # Отображение в окне
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img_pil)
    label.imgtk = imgtk
    label.config(image=imgtk)

    root.after(20, update_frame)


# Запуск
print("Запуск через 3 секунды... Переключитесь на Chrome Dino")
time.sleep(3)
pyautogui.press('space')
game_start_time = time.time()
update_frame()
root.mainloop()