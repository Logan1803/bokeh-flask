#import dependencies
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
# import skimage as ski

#download MiDaS model
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')
midas.eval()

#input tranformational pipeline
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

# #Hook into opencv
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     ret, frame = cap.read()

#     cv2.imshow('CV2Frame', frame)

#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         cap.release()
#         cv2.destroyAllWindows()

img = cv2.imread('anish_mee.jpg')

resize_percentage = 30
width = int(img.shape[1] * resize_percentage / 100)
height = int(img.shape[0] * resize_percentage / 100)
print(width, height)
img = cv2.resize(img, (width, height))

cv2.imshow('Given image', img)


#Tranform image to input for midas
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgbatch = transform(img).to('cpu')

#make a prediction
with torch.no_grad():
    prediction = midas(imgbatch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1), 
        size = img.shape[:2], 
        mode = 'bicubic', 
        align_corners = False
    ).squeeze()
    
    output = prediction.to('cpu').numpy()
    output_norm = cv2.normalize(output, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # print(output_norm)
    # cv2.imshow("depth_img", output_norm)
    # plt.imshow(output_norm)

ret,mask = cv2.threshold(output_norm,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
mask_fore = cv2.bitwise_not(mask)

# cv2.imshow('thresholded image', mask)

bokeh_image = np.copy(img)
foreground_sub = np.copy(img)
foreground_sub = cv2.bitwise_and(foreground_sub, foreground_sub, mask = mask_fore)
bokeh_image = cv2.bitwise_and(bokeh_image, bokeh_image, mask = mask)



bokeh_image = cv2.GaussianBlur(bokeh_image,(5,5),5)
# cv2.imshow('background', bokeh_image)
# cv2.imshow('foreground', foreground_sub)

bokeh_image = cv2.add(bokeh_image, foreground_sub)
# img_para = list(img.shape[:2])
# print(img_para)
# print(bokeh_image[0][0])
# blur = np.zeros(img_para)
# for i in range(height):
#     for j in range(width):
#         if not(bokeh_image[i][j].all() == 0 or foreground_sub[i][j].all() == 0):
#             blur[i][j] = 1
#         bokeh_image[i][j] += foreground_sub[i][j]

# for i in range(height):
#     for j in range(width):
#         if blur[i][j] == 1:


bokeh_image = cv2.cvtColor(bokeh_image, cv2.COLOR_BGR2RGB)
cv2.imshow("Bokeh image", bokeh_image)

# plt.show()

cv2.waitKey(0)

cv2.destroyAllWindows()

