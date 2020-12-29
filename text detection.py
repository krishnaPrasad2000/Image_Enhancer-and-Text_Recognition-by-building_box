import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\--FILE PATH---\tesseract.exe'

img=cv2.imread(r"C:\Users\kp473\OneDrive\Pictures\Screenshots\Screenshot (19).png",1)
img=cv2.resize(img,(680,480))
rows,cols,ch =img.shape
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# boxes=pytesseract.image_to_boxes(img)
# for b in boxes.splitlines():
#     b = b.split(' ')
#     print(b)
#     x,y,w,h=int(b[1]),int(b[2]),int(b[3]),int(b[4])
#     cv2.rectangle(img,(x,rows-y),(w,rows-h),(0,255,0),1)

# print(pytesseract.image_to_string(img))
n=input('Enter the string : ')
boxes=pytesseract.image_to_data(img)
for y,b in enumerate(boxes.splitlines()):
  if y!=0:
    b = b.split()
    print(b)
    if len(b)==12 and b[11]==n:
      x,y,w,h=int(b[6]),int(b[7]),int(b[8]),int(b[9])
      cv2.rectangle(img,(x,y),(w+x,h+y),(0,0,255),2)

print(boxes)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
