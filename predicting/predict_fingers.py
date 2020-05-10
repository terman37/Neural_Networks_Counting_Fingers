# imports
import cv2
import copy
import os
import pickle
import numpy as np

from keras.models import load_model


def main():
    # init variables
    x, y, w, h = 400, 150, 200, 200
    x2, y2, w2, h2 = 10, 370, 100, 100
    wfinal, hfinal = 100, 100
    key = 0
    model_path = '../models'

    # Load saved models
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    cnn_model = load_model(os.path.join(model_path, 'model_cnn.h5'))

    cnn_preproc_model = load_model(os.path.join(model_path, 'model_cnn_preproc.h5'))

    ml_model = pickle.load(open(os.path.join(model_path, 'model_classic.sav'), 'rb'))
    pca_model = pickle.load(open(os.path.join(model_path, 'model_classic_pca.sav'), 'rb'))

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

        # Image to predict
        to_predict = cv2.resize(roi, (wfinal, hfinal))
        to_predict = cv2.cvtColor(to_predict, cv2.COLOR_BGR2GRAY)

        key = cv2.waitKey(5) & 0xff

        # predict using ml
        myimg = to_predict #/ 255.
        myimg = myimg.reshape(1, wfinal * hfinal)
        myimg_pca = pca_model.transform(myimg)
        result = ml_model.predict(myimg_pca)
        pred = result[0]
        cv2.putText(window, "ml model : %d" % pred, (397, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # predict using cnn
        myimg = to_predict.reshape(1, hfinal, wfinal, 1)
        myclass = cnn_model.predict(myimg)
        pred = np.argmax(myclass)
        cv2.putText(window, "cnn model : %d" % pred, (380, 380), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # predict using cnn with pre processing
        myimg = cv2.GaussianBlur(to_predict, ksize=(1, 1), sigmaX=1000).astype('uint8')
        myimg = cv2.adaptiveThreshold(myimg, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      thresholdType=cv2.THRESH_BINARY, blockSize=5, C=3)
        myimg = myimg.reshape(1, hfinal, wfinal, 1)
        myclass = cnn_preproc_model.predict(myimg)
        pred = np.argmax(myclass)
        cv2.putText(window, "cnn preproc : %d" % pred, (380, 410), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 128, 0), 2)
        window[y2:y2 + h2, x2:x2 + w2] = cv2.cvtColor(myimg.reshape(hfinal, wfinal, 1), cv2.COLOR_GRAY2BGR)

        # Display image
        cv2.imshow('WebCam', window)

    # Release webcam
    cam.release()


if __name__ == "__main__":
    main()
