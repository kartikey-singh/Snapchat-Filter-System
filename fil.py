import numpy as np
import cv2

def nothing(x):
    pass

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
# filters = ['Filters/cat_filter.png','Filters/citybg_filter.png','Filters/dog1_filter.png','Filters/dog_filter.png','Filters/hat_filter.png','Filters/rabbit1_filter.png','Filters/rabbit_filter.png','Filters/rainbow_filter.png','Filters/unicorn_filter.png']
filters = ['Filters/cat_filter.png','Filters/citybg_filter.png','Filters/dog1_filter.png','Filters/dog_filter.png','Filters/hat1_filter.png','Filters/rabbit1_filter.png','Filters/rabbit_filter.png','Filters/rainbow_filter.png','Filters/unicorn_filter.png','Filters/spectacles2.png']
cv2.namedWindow('Snapchat Filters')
cv2.createTrackbar('Filter','Snapchat Filters',0,9,nothing)

while 1:
    pos = cv2.getTrackbarPos('Filter','Snapchat Filters')
    img2 = cv2.imread(filters[pos])
    ret, img = cap.read()
    row1,col1,p1 = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    roi_gray =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    for (x,y,w,h) in faces:
        if pos == 4 or pos == 8:
            if y > h :
                img2 = cv2.resize(img2, (w,h))
            else :
                img2 = cv2.resize(img2, (w,y-2))   
        elif pos == 1:
            img2 = cv2.resize(img2, (col1,row1))        
        elif pos == 7:
            if y + h + h//2 > row1 :
                img2 = cv2.resize(img2, (w,row1 - y - h//2))
            else:
                img2 = cv2.resize(img2, (w,h))
        elif pos == 9:
            img2 = cv2.resize(img2,(w,h//2 + h//4))        
        else :  
            img2 = cv2.resize(img2, (w,h))

        # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        
        rows,cols,channels = img2.shape
        
        if pos == 4 or pos == 8:
            roi = img[y-rows:y ,x:x+cols]     
        elif pos == 1:
            roi = img[0:row1,0:col1]      
        elif pos == 7:
            if y + rows + rows//2 >  row1 :
                roi = img[y + rows//2: row1 ,x:x+cols]
            else :        
                roi = img[y + rows//2:y + rows + rows//2 ,x:x+cols]       
        elif pos == 9:
            roi = img[y:y + rows,x: x + cols]                          
        else:
            roi = img[y:y+rows ,x:x+cols]       

        img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
        
        dst = cv2.add(img1_bg,img2_fg)
        if pos == 4 or pos == 8:
            img[y-rows:y, x:x+cols] = dst
        elif pos == 1:
            img[0:rows,0:cols] = dst    
        elif pos == 7:
            if y + rows + rows//2 >  row1 :
                img[ y+ rows//2:row1, x:x+cols] = dst
            else :    
                img[ y+ rows//2:y+rows + rows//2, x:x+cols] = dst   
        elif pos == 9:
            img[y:y + rows,x: x + cols] = dst                        
        else:
            img[ y:y+rows, x:x+cols] = dst  
        
        # roi_color = img[y:y+h, x:x+w]
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eyes:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('Snapchat Filters',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()