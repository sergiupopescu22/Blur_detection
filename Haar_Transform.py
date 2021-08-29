import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

image = cv2.imread('poza_2_blur.jpg',cv2.IMREAD_UNCHANGED) #read the input image
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#transform it from rgb to greyscale

image_to_be_displayed= image

# resize image
height, width= image_to_be_displayed.shape
if height >= 800 :
    dim = (width, 800)
    # resize image
    image_to_be_displayed = cv2.resize(image_to_be_displayed, dim, interpolation=cv2.INTER_AREA)#resize it in order to fit on the screen
    #cv2.imshow('resized', image_to_be_displayed)

height, width= image_to_be_displayed.shape
if width >= 800 :
    dim = (800, height)
    # resize image
    image_to_be_displayed = cv2.resize(image_to_be_displayed, dim, interpolation=cv2.INTER_AREA)#resize it in order to fit on the screen
    #cv2.imshow('resized',image_to_be_displayed )



M, N = image.shape
# Crop input image to be 3 divisible by 2
image = image[0:int(M / 16) * 16, 0:int(N / 16) * 16]
#cv2.imshow('Original_Image',image)

# Convert to float for more resolution for use with pywt
image_to_work_with = np.float64(image)
image_to_work_with /= 255

cA1, (cH1, cV1, cD1) = pywt.dwt2(image_to_work_with, 'haar')#haar transform lv1
cA2, (cH2, cV2, cD2) = pywt.dwt2(cA1, 'haar')#haar transform lv2
cA3, (cH3, cV3, cD3) = pywt.dwt2(cA2, 'haar')#haar transform lv3

#create the haar transoform visualisation for each level
img_concate_Hori1=np.concatenate((cA1,cH1),axis=1)
img_concate_Hori2=np.concatenate((cV1,cD1),axis=1)
img_concate_Verti=np.concatenate((img_concate_Hori1,img_concate_Hori2),axis=0)
#cv2.imshow('Haar_transform1',img_concate_Verti)

img_concate_Hori1=np.concatenate((cA2,cH2),axis=1)
img_concate_Hori2=np.concatenate((cV2,cD2),axis=1)
img_concate_Verti=np.concatenate((img_concate_Hori1,img_concate_Hori2),axis=0)
#cv2.imshow('Haar_transform2',img_concate_Verti)

img_concate_Hori1=np.concatenate((cA3,cH3),axis=1)
img_concate_Hori2=np.concatenate((cV3,cD3),axis=1)
img_concate_Verti=np.concatenate((img_concate_Hori1,img_concate_Hori2),axis=0)
#cv2.imshow('Haar_transform3',img_concate_Verti)

#STEP 2----------------------------------------------------------------------------

# Construct the edge map in each scale Step 2
E1 = np.sqrt(np.power(cH1, 2) + np.power(cV1, 2) + np.power(cD1, 2))
E2 = np.sqrt(np.power(cH2, 2) + np.power(cV2, 2) + np.power(cD2, 2))
E3 = np.sqrt(np.power(cH3, 2) + np.power(cV3, 2) + np.power(cD3, 2))

cv2.imshow('E1',E1)
cv2.imshow('E2',E2)
cv2.imshow('E3',E3)

#STEP 3----------------------------------------------------------------------------
M1, N1 = E1.shape

# Sliding window size level 1
sizeM1 = 8
sizeN1 = 8

# Sliding windows size level 2
sizeM2 = 4
sizeN2 = 4

# Sliding windows size level 3
sizeM3 = 2
sizeN3 = 2

# Number of edge maps, related to sliding windows size
N_iter = int((M1 / sizeM1) * (N1 / sizeN1))

Emax1 = np.zeros((N_iter))
Emax2 = np.zeros((N_iter))#this array has the same value as Emax1 because even if the image is samller, the edge map is also samller
Emax3 = np.zeros((N_iter))#this array has the same value as Emax1 because even if the image is samller, the edge map is also samller

count = 0

# Sliding windows index of level 1
x1 = 0
y1 = 0
# Sliding windows index of level 2
x2 = 0
y2 = 0
# Sliding windows index of level 3
x3 = 0
y3 = 0

# Sliding windows limit on horizontal dimension
Y_limit = N1 - sizeN1

while count < N_iter:
    # Get the maximum value of slicing windows over edge maps
    # in each level
    Emax1[count] = np.max(E1[x1:x1 + sizeM1, y1:y1 + sizeN1])
    Emax2[count] = np.max(E2[x2:x2 + sizeM2, y2:y2 + sizeN2])
    Emax3[count] = np.max(E3[x3:x3 + sizeM3, y3:y3 + sizeN3])

    # if sliding windows ends horizontal direction
    # move along vertical direction and resets horizontal
    # direction
    if y1 == Y_limit:
        x1 = x1 + sizeM1
        y1 = 0

        x2 = x2 + sizeM2
        y2 = 0

        x3 = x3 + sizeM3
        y3 = 0

        count += 1

    # windows moves along horizontal dimension
    else:

        y1 = y1 + sizeN1
        y2 = y2 + sizeN2
        y3 = y3 + sizeN3
        count += 1

threshold = 35/255 #35 is the  recommanded value in the paper

#calculate bool arrays depending on their values(compared with the threshold)
EdgePoint1 = Emax1 > threshold;
EdgePoint2 = Emax2 > threshold;
EdgePoint3 = Emax3 > threshold;

n_edges = EdgePoint1.shape[0]
print('Number of values: ',n_edges)

#RULE 1: EdgePointx is type bool so + means OR
EdgePoint = EdgePoint1 + EdgePoint2 + EdgePoint3

NumberOfEdgePoints = 0

for i in EdgePoint:
    if i == True:
        NumberOfEdgePoints+=1

print('Number of edge points: ',NumberOfEdgePoints)

#RULE 2:
DAstructure = (Emax1 > Emax2) * (Emax2 > Emax3);

NumberOfDA = 0

for i in DAstructure:
    if i == True:
        NumberOfDA+=1

print('Number of Dirac and A Step points: ',NumberOfDA)

# Rule 3 Roof-Structure or Gstep-Structure

RGstructure = np.zeros((n_edges))

for i in range(n_edges):
    if EdgePoint[i] == True:
        if Emax1[i] < Emax2[i] and Emax2[i] < Emax3[i]:
            RGstructure[i] = 1

NumberOfRG=0;
for i in RGstructure:
    if i == 1:
        NumberOfRG+=1;

print('Number of Roof and G step points: ',NumberOfRG)

# Rule 4 Roof-Structure

RSstructure = np.zeros((n_edges))

for i in range(n_edges):
    if EdgePoint[i] == True:
        if Emax2[i] > Emax1[i] and Emax2[i] > Emax3[i]:
            RSstructure[i] = 1

NumberOfR=0;
for i in RSstructure:
    if i == 1:
        NumberOfR+=1;

print('Number of Roof points: ',NumberOfR)


# Rule 5 Edge more likely to be in a blurred image

BlurC = np.zeros((n_edges));

for i in range(n_edges):
    if RGstructure[i] == 1 or RSstructure[i] == 1:
        if Emax1[i] < threshold:
            BlurC[i] = 1

NumberOfBlur=0;
for i in BlurC:
    if i == 1:
        NumberOfBlur+=1;

print('Number of Blur points: ',NumberOfBlur)

# Step 6
Per = NumberOfDA / NumberOfEdgePoints

# Step 7
if (NumberOfRG + NumberOfR) == 0:

    BlurExtent = 100
else:
    BlurExtent = NumberOfBlur / (NumberOfRG + NumberOfR)


print('Per: ',Per)
print('BlurExtent: ',BlurExtent)

result=''
if BlurExtent > 0.8:
    print('THE IMAGE IS BLURRED')
    result='blurred'

else:
    result='not blurred'



font = cv2.FONT_HERSHEY_SIMPLEX

text = 'Image Status: ' + result;
cv2.putText(image_to_be_displayed,text,(1,30), font, 1,(255,255,255),2)

cv2.imshow('Original_Image',image_to_be_displayed)


cv2.waitKey(0)



