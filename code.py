import cv2
import numpy as np
from numpy.fft import fftshift
import matplotlib.pyplot as plt

img = cv2.imread('camera.jpg', 0)
#img = cv2.imread('noiseIm.jpg',0)  #Q4 only
img2 = cv2.imread('denoiseIm.jpg',0) 
# img = np.zeros((512,512))
# for i in range(512):
#     img[256,i] = 1
ip_xlen = len(img)
ip_ylen = len(img[0])
ip_size = ip_xlen * ip_ylen

def dist(c1,c2):
    return ((c1[0]-c2[0])**2+(c1[1]-c2[1])**2)**(1/2)

def get_BW_filter(xlen,ylen,cutoff,n):
    filter = np.zeros((xlen,ylen),dtype=float)
    center = [xlen/2,ylen/2]
    for i in range(xlen):
        for j in range(ylen):
            r = dist([i,j],center)
            frac = (r/cutoff)**(2*n)
            filter[i][j] = 1/(1+frac)
    return filter

def get_ideal_filter(xlen,ylen,cutoff):
    filter = np.zeros((xlen,ylen),dtype=float)
    center = [xlen/2,ylen/2]
    for i in range(xlen):
        for j in range(ylen):
            r = dist([i,j],center)
            if r<=cutoff:
                filter[i][j]=1
    return filter

def get_gaussian_filter(xlen,ylen,cutoff):
    filter = np.zeros((xlen,ylen),dtype=float)
    center = [xlen/2,ylen/2]
    for i in range(xlen):
        for j in range(ylen):
            r = dist([i,j],center)
            filter[i][j] = np.exp(-1*((r**2)/(2*(cutoff**2))))
    return filter

def getDenoiseFilter(cutoff):
    filter = np.zeros((2*ip_xlen,2*ip_ylen),dtype=float)
    xlen = 2*ip_xlen
    ylen = 2*ip_ylen
    center1 = [192,192]
    center2 = [320,320]
    for i in range(xlen):
        for j in range(ylen):
            r1 = dist([i,j],center1)
            r2 = dist([i,j],center2)
            # if r1<=cutoff or r2 <= cutoff: #ideal
            #     filter[i][j]=1
            frac1 = (r1/cutoff)**(2*2)
            frac2 = (r2/cutoff)**(2*2)
            filter[i][j] = 1/(1+frac1) + 1 /(1+frac2)
            #filter[i][j] = np.exp(-1*((r1**2)/(2*(cutoff**2)))) + np.exp(-1*((r2**2)/(2*(cutoff**2)))) #gaussian
    return (1-filter)

def zero_pad(img,xlenAdd,ylenAdd):
    xlen = len(img)
    ylen = len(img[0])
    padded = np.zeros((xlen+xlenAdd, ylen+ylenAdd))
    for i in range(xlen):
        for j in range(ylen):
            padded[i][j] = img[i][j]
    return padded

def getDft(img):
    dft = np.fft.fft2(img)
    return dft

def getCentDft(img):
    xlen = len(img)
    ylen = len(img[0])
    centered_img = np.zeros((xlen,ylen),dtype=complex)
    for i in range(xlen):
        for j in range(ylen):
            centered_img[i][j] = img[i][j]*((-1)**(i+j))
    dft = np.fft.fft2(centered_img)
    return dft

def getIDFT(img):
    ift = np.fft.ifft2(img)
    return ift

def element_product(img1,img2):
    xlen = len(img1)
    ylen = len(img1[0])
    prod = np.zeros((xlen,ylen),dtype=complex)
    for i in range(xlen):
        for j in range(ylen):
            prod[i][j] = img1[i][j]*img2[i][j]
    return prod   

def postProcess1(img):
    xlen = len(img)
    ylen = len(img[0])
    fin= np.zeros((ip_xlen,ip_ylen),dtype=int)
    for i in range(ip_xlen):
        for j in range(ip_ylen):
            num = img[i][j].real*((-1)**(i+j))
            round(num)
            fin[i][j] = int(num)
    return fin

def postProcess2(img):
    fin = np.zeros((ip_xlen,ip_ylen))
    for i in range(ip_xlen):
        for j in range(ip_ylen):
            fin[i][j] = img[i+4][j+4].real
    return fin

def showDftLog(dft,name):
    mat = abs(dft)
    fin = np.log(1+mat)/np.log(1+np.max(mat))
    cv2.imshow(name,fin/np.max(fin))

def showDftRaw(dft,name):
    mat = abs(dft)
    mat = 100*(mat - np.min(mat))/(np.max(mat)-(np.min(mat)))  #for q4 multiply by 255
    cv2.imshow(name,mat)
    if name == "noisyCentDft_Raw":
        cv2.imwrite("Centered_DFT_noiseIm.jpg",255*mat) 

# def showFilter(filter,name):
#     spatialFilter = getIDFT(filter)
#     spatialFilter = spatialFilter.real
#     print(spatialFilter)
#     spatialFilter = np.round((spatialFilter-np.min(spatialFilter))/(np.max(spatialFilter)-np.min(spatialFilter)))
#     cv2.imshow(name,fftshift(spatialFilter))


#Q1

pad = zero_pad(img,ip_xlen,ip_ylen)
cv2.imshow("img",img)
cv2.imshow("paddedimg",pad/255)
ip_cent_dft = getCentDft(pad)
ip_dft = getDft(pad)

showDftLog(ip_dft,"dft")    
showDftRaw(ip_dft,"dft_raw")
showDftLog(ip_cent_dft,"dft_cent")
showDftRaw(ip_cent_dft,"dft_cent_raw")

filter10 = get_BW_filter(2*ip_xlen,2*ip_ylen,10,2)
filter30 = get_BW_filter(2*ip_xlen,2*ip_ylen,30,2)
filter60 = get_BW_filter(2*ip_xlen,2*ip_ylen,60,2)


cv2.imshow("filter10", filter10)
cv2.imshow("filter30", filter30)
cv2.imshow("filter60", filter60)

product10 = element_product(ip_cent_dft,filter10)
centblurred10 = getIDFT(product10)
blurred10 = postProcess1(centblurred10)
cv2.imshow("blurred10",blurred10/255)

product30 = element_product(ip_cent_dft,filter30)
centblurred30 = getIDFT(product30)
blurred30 = postProcess1(centblurred30)
cv2.imshow("blurred30",blurred30/255)

product60 = element_product(ip_cent_dft,filter60)
centblurred60 = getIDFT(product60)
blurred60 = postProcess1(centblurred60)
cv2.imshow("blurred60",blurred60/255)

# showFilter(filter30,"spatialFilter30")
# showFilter(filter10,"spatialFilter10")
# showFilter(filter60,"spatialFilter60")


# filter200 = get_BW_filter(2*ip_xlen,2*ip_ylen,200,2)
# product200 = element_product(ip_cent_dft,filter200)
# centblurred200 = getIDFT(product200)
# blurred200 = postProcess1(centblurred200)
# cv2.imshow("blurred200",blurred200/255)

# #Q3

# boxFilter = np.ones((9,9))/81

# paddedFilter = zero_pad(boxFilter,ip_xlen-1,ip_ylen-1)


# paddedImg = zero_pad(img,8,8)

# print(paddedFilter)

# cv2.imshow("paddedFilter", paddedFilter)
# cv2.imshow("paddedImage",paddedImg/255)

# # filtDft = getCentDft(paddedFilter)
# # imgDft = getCentDft(paddedImg)

# filtDft = getDft(paddedFilter)
# imgDft = getDft(paddedImg)


# showDftLog(filtDft,"dftFilter")
# showDftLog(imgDft,"dftImage")

# product = element_product(filtDft,imgDft)

# blurred = (getIDFT(product))
# blurred = postProcess2(blurred)

# cv2.imshow("blurred image",blurred/255)

# cvBlurred = cv2.boxFilter(img,-1,(9,9))

# cv2.imshow("Blurred using cv2", cvBlurred)

# #Q4 
# cv2.imshow("NoisyImg",img)
# #band = 1 - (get_ideal_filter(2*ip_xlen,2*ip_ylen,90) - get_ideal_filter(2*ip_xlen,2*ip_ylen,90))
# #band = 1 - (get_BW_filter(2*ip_xlen,2*ip_ylen,130,2) - get_BW_filter(2*ip_xlen,2*ip_ylen,60,2))
# #band = 1 - (get_gaussian_filter(2*ip_xlen,2*ip_ylen,120) - get_gaussian_filter(2*ip_xlen,2*ip_ylen,40))
# band = getDenoiseFilter(30)
# showDftLog(band,"BandDft")
# pad = zero_pad(img,ip_xlen,ip_ylen)
# cent_dft = getCentDft(pad)
# dft2 = getCentDft(img2)
# dft3 = getCentDft(img)
# showDftLog(cent_dft,"noisyCentDft")
# showDftRaw(cent_dft,"noisyCentDft_Raw")
# product = element_product(cent_dft,band)
# denoise = getIDFT(product)
# denoise = postProcess1(denoise)
# fin_dft = getCentDft(denoise)
# showDftLog(fin_dft,"denoiseDft")
# showDftRaw(fin_dft/255,"denoiseDft_raw")
# cv2.imshow("Original Image",denoise/255)
# cv2.imwrite("Restored.jpg",denoise)

# cv2.imshow('lineIMG',img)
# dft = getDft(img)
# centdft = getCentDft(img)
# showDftLog(dft,"dft")
# showDftLog(centdft,"centered dft")

# for i in range(len(dft)):
#     print(dft[i][0])

cv2.waitKey(0)
cv2.destroyAllWindows()
