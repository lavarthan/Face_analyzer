import cv2

# for teeth detection we used smile detection xml with high accuracy
cascade_face = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
cascade_smile = cv2.CascadeClassifier('models/haarcascade_smile.xml')


# for detect teeth
def detect(img, percentage, frame):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # if you want change the quality percentage change here
    if percentage > 50:
        quality = " Good Quality"
    else:
        quality = "Low Quality"
    x = frame[0][0]
    y = frame[0][1]
    w = frame[1][0]
    h = frame[1][1]
    face = [(x, y, w, h)]
    face = cascade_face.detectMultiScale(gray, 1.3, 5)

    for (x_face, y_face, w_face, h_face) in face:
        ri_grayscale = gray[y_face:y_face + h_face, x_face:x_face + w_face]
        ri_color = img[y_face:y_face + h_face, x_face:x_face + w_face]
        smile = cascade_smile.detectMultiScale(ri_grayscale, 1.7, 25)
        if len(smile) != 0:
            for (x_smile, y_smile, w_smile, h_smile) in smile:
                cv2.rectangle(ri_color, (x_smile, y_smile), (x_smile + w_smile, y_smile + h_smile), (255, 0, 130), 2)
                cv2.putText(ri_color, '', (x_smile + w_smile, y_smile + h_smile), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (148, 0, 211), 1, cv2.LINE_AA)
                return (x_smile, y_smile), (x_smile + w_smile, y_smile + h_smile), quality, "yes"
        else:
            return (0, 0), (0, 0), quality, "no"
