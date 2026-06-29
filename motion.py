import cv2


def count_motion_area(fgmask, min_area):
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = 0.0
    motion_detected = False
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        total_area += area
        motion_detected = True
    return motion_detected, total_area
