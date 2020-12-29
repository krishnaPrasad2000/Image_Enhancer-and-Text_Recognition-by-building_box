import cv2
import numpy as np
from matplotlib import pyplot as plt
import math as m
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\----FILE PATH-----\tesseract.exe'


while True:

   img=cv2.imread("FILE PATH",1)

   img=cv2.resize(img,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
  
   rows,cols,channels=img.shape
   
   black=np.zeros([rows,cols,3],np.int8)
   img_grayscale=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   img_G_blur=cv2.GaussianBlur(img_grayscale,(5,5),1)

   thresh=cv2.Canny(img_G_blur,190,180) #### threshold 1,2 can be done by using trackbar
   
   kernal=np.ones((4,4),np.uint8)
   dil_image=cv2.dilate(thresh,kernal,iterations=2)
   e_image=cv2.erode(dil_image,kernal,iterations=1)

   contours,heirarchy=cv2.findContours(e_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
   
   for i in contours:
        area=cv2.contourArea(i)
        print(area)
        perimeter=cv2.arcLength(i,True)
        approx=cv2.approxPolyDP(i,0.1*perimeter,True)
         
###################################################################   four cornor points.
        leftmost = tuple(i[i[:,:,0].argmin()][0]) 
        rightmost = tuple(i[i[:,:,0].argmax()][0]) 
        topmost = tuple(i[i[:,:,1].argmin()][0]) 
        bottommost = tuple(i[i[:,:,1].argmax()][0])

        if area>5000:
           if len(approx)==4 :
              cv2.drawContours(img,contours,-1,(0,255,0),6)

              cv2.circle(img,leftmost, 8, (0, 0, 255), -1)
              cv2.circle(img, rightmost, 8, (0, 0, 255), -1)
              cv2.circle(img, topmost, 8, (0, 0, 255), -1)
              cv2.circle(img, bottommost, 8, (0, 0, 255), -1)
  
        M=cv2.moments(i)                            ##################### center of the contour
        cx = int(M['m10']/M['m00']) 
        cy = int(M['m01']/M['m00'])

############################################################ in which orientation to rotate
        if topmost[0] > bottommost[0] :
           a=leftmost[1]-topmost[1]
           b=topmost[0]-leftmost[0]
           c=leftmost[1]-topmost[1]
           d=m.sqrt(b^2 + c^2)
           result=a/d
           angle= -m.atan(result)

           pts1=np.float32([[leftmost[0]+20,leftmost[1]],[topmost[0]+30,topmost[1]],
               [bottommost[0],bottommost[1]],[rightmost[0],rightmost[1]]])

        elif topmost[0]<bottommost[0]:

           a=rightmost[1]-topmost[1]
           b=rightmost[0]-topmost[0]
           c=rightmost[1]-topmost[1]
           d=m.sqrt(b^2 + c^2)
           result=a/d
           angle= m.atan(result)

           pts1=np.float32([[leftmost[0],leftmost[1]],[topmost[0],topmost[1]],
               [bottommost[0]+20,bottommost[1]],[rightmost[0],rightmost[1]]])

        else:
            angle=0   

       
   ########################################################## alligning the page using warpPerspective
   R=cv2.getRotationMatrix2D((cx,cy),angle,1)
   img_resized=cv2.warpAffine(img,R,(cols,rows))

   pts2=np.float32([[0,0],[rows,0],[0,cols],[rows,cols]])
   N=cv2.getPerspectiveTransform(pts1,pts2)
   dst=cv2.warpPerspective(img_resized,N,(rows,cols))
   final=cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
  
   _, th1=cv2.threshold(final,55,255,cv2.THRESH_BINARY)
   kernal=np.ones((1,1),np.uint8)


   th2=cv2.adaptiveThreshold(final,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,91, 10)

   th2=cv2.medianBlur(th2,1)
   th2=cv2.dilate(th2,kernal,iterations=7)
   
   #############################################################  creation of bounding box

   th2=cv2.cvtColor(th2,cv2.COLOR_BGR2RGB)

   n=input('Enter the string : ')

   boxes=pytesseract.image_to_data(th2)

   for y,b in enumerate(boxes.splitlines()):

      if y!=0:
         b = b.split()
   
         if len(b)==12 and b[11]==n:

            x,y,w,h=int(b[6]),int(b[7]),int(b[8]),int(b[9])
            cv2.rectangle(th2,(x,y),(w+x,h+y),(255,0,0),2)

   cv2.imshow('th2',th2)


   if cv2.waitKey(0) & 0xFF==ord('q'):

           break


cv2.destroyAllWindows()
