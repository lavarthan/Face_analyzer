import cv2


def blur_background(img, f_percentage, frame):
    x = frame[0][0]
    y = frame[0][1]
    w = frame[1][0] - x
    h = frame[1][1] - y
    faces = [(x, y, w, h)]

    # if you want to chance the percentage and blur change here
    if f_percentage > 50:
        for (x, y, w, h) in faces:
            face = img[y:y + h, x:x + w]
            frame = cv2.blur(img, ksize=(15, 15))
            frame[y:y + h, x:x + w] = face
            return frame
    return img
