import cv2
from Final.blur import blur_background
from Final.teeth import detect
from Final.percentage import face_percentage
from Final.emotion import emotion

while True:
    while True:
        fname = input('enter file')
        fname = 'C:\Extra\Face_Recognition\Final\input_files'+'\\'+fname

        # read the image
        img = cv2.imread(fname)

        # get the face size
        emotions = emotion(img)
        frame = (emotions[0], emotions[1])
        print(fname)

        # get the face percentage
        percentage = face_percentage(img, frame)

        # blur the image if we have to
        blured = blur_background(img, percentage, frame)
        # cv2.imshow("blur", blured)
        # cv2.waitKey()

        # detect the teeth
        teeth = detect(img, percentage, frame)

        # get the quality of the photo
        quality = teeth[2]
        t_x = teeth[0]
        t_y = teeth[1]

        x = emotions[0]
        y = emotions[1]

        # get the emotion in the face
        sentiment = emotions[2]

        # draw rectangle and put appropriate texts

        cv2.rectangle(blured, x, y, (255, 0, 0), 2)
        cv2.putText(blured, sentiment, (y[0], y[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (148, 0, 211), 2, cv2.LINE_AA)
        cv2.putText(blured, quality.split(" ")[0], (y[0], y[1] - 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (148, 0, 211), 2,
                    cv2.LINE_AA)
        cv2.putText(blured, quality.split(" ")[1], (y[0], y[1] - 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (148, 0, 211), 2,
                    cv2.LINE_AA)

        if teeth[3] == "yes":
            cv2.putText(blured, 'Teeth', (y[0], y[1] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (148, 0, 211), 2, cv2.LINE_AA)
            cv2.putText(blured, 'detect', (y[0], y[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (148, 0, 211), 2, cv2.LINE_AA)
        else:
            cv2.putText(blured, 'No', (y[0], y[1] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (148, 0, 211), 2, cv2.LINE_AA)
            cv2.putText(blured, 'teeth', (y[0], y[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (148, 0, 211), 2, cv2.LINE_AA)

        # display the image
        cv2.imshow(fname, blured)
        cv2.waitKey()

        # save the image
        # give proper name and the correct directory to avoid the errors
        cv2.imwrite(fname.split('.')[-1] + '-tested.jpg', blured)
        print("saved")
    # except Exception as e:
    #     print(e)
    #     print('Something wrong! Possible reasons...\n1. Photo quality is very low\n2. Photo size is very low\n3. Two '
    #           'or more faces in the photo\n4. check the input')
    # input("press enter")
