
import cv2
import copy
from IPython import display
import matplotlib.pyplot as plt


cam = cv2.VideoCapture(0)
# cv2.namedWindow('WebCam', cv2.WINDOW_NORMAL)
while True:
    # Get image from Camera
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    
    plt.axis('off')
    plt.title("Input Stream")
    plt.imshow(frame)
    plt.show()
    
    display.clear_output(wait=True)
    # copy frame to window to work on the copy
#     window = copy.deepcopy(frame)
#     cv2.imshow('WebCam', window)
#     IPython.display.Image(window)
    
cam.release()