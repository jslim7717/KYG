import matplotlib.pyplot as plt
import cv2
import numpy as np

import pydicom as dicom

ds = dicom.dcmread('sample.dcm')


if 'PatientName' in ds:

    ds.PatientName = ''

    
ds.save_as('sample_2.dcm')

ds_img = ds.pixel_array

gt_img = cv2.imread('sample.png', cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(ds_img, cmap=plt.cm.gray)
plt.title('DICOM Image')

plt.subplot(1, 2, 2)
plt.imshow(gt_img, cmap=plt.cm.gray)
plt.title('ground truth')

plt.show()