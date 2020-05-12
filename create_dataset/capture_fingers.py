# imports
import cv2
import copy
import os
import re


def main():
    # init variables
    x, y, w, h = 400, 150, 200, 200
    x2, y2, w2, h2 = 10, 370, 100, 100
    nb = -1
    key = 0
    save_path = '../data/originals'
    last_saved = 0

    # Define last count value for original pictures
    count = len(os.listdir(save_path)) + 1
    if count > 1:
        maxid = max([int(re.findall("[0-9]+", i)[1]) for i in os.listdir(save_path)])
    else:
        maxid = 1

    # Get Webcam stream
    cam = cv2.VideoCapture(0)
    cv2.namedWindow('WebCam', cv2.WINDOW_NORMAL)

    while key != ord('q'):
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)
        window = copy.deepcopy(frame)

        # Define a ROI, show bounds as a rectangle
        roi = frame[y:y + h, x:x + w]
        cv2.rectangle(img=window, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)

        # Define a frame for last saved picture
        if nb != -1:
            cv2.rectangle(img=window, pt1=(x2, y2), pt2=(x2 + w2, y2 + h2), color=(0, 0, 255), thickness=2)
            window[y2:y2 + h2, x2:x2 + w2] = last_saved
            cv2.putText(window, "%d" % nb, (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

        # Use keyboard 
        key = cv2.waitKey(5) & 0xff

        # select nb of finger for saving
        cv2.putText(window, "Press 0,1,2,3,4,5 to save. (q to quit)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2)
        cv2.putText(window, "nb images: %d" % (count - 1), (360, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        keys = {ord('0'): 0, ord('1'): 1, ord('2'): 2, ord('3'): 3, ord('4'): 4, ord('5'): 5}
        if key in keys:
            nb = keys.get(key)
            # save image
            filename = os.path.join(save_path, '%d_original_%d.png' % (nb, maxid + 1))
            cv2.imwrite(filename, roi)
            # last saved image for display
            last_saved = cv2.resize(roi, (100, 100))
            maxid += 1
            count += 1
            key = 0

        # Display image
        cv2.imshow('WebCam', window)

    # Release webcam
    cam.release()


if __name__ == "__main__":
    main()
