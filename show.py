import cv2
import numpy as np
a = cv2.imread("Frame_08.bmp", 0)
b = np.fromfile("h_Asal.bin", dtype=np.uint8).reshape(a.shape)
cv2.imshow("test", a*255); cv2.waitKey()
cv2.imshow("test2", b*255); cv2.waitKey()
cv2.waitKey()

