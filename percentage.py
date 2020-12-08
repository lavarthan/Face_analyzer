# to get the percentage of face in a whole photo
# note :- hair is not in the face


def face_percentage(img, frame):
    x = frame[0][0]
    y = frame[0][1]
    w = frame[1][0] - x
    h = frame[1][1] - y
    height, width, channels = img.shape
    total_image = height * width
    f_total = h * w
    percentage = round((f_total / total_image) * 100, 2)
    return percentage
