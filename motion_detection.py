import cv2
import time
import imutils

cam=cv2.VideoCapture(0)
time.sleep(1)

fgbg=cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=50,detectShadows=True)

first_frame=None
area=1000
last_update=time.time()

while True:
    _,img = cam.read()
    text="Normal"
    img=imutils.resize(img,width=500)
    
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    grayImg = hsv[:,:,2]

    gussianImg = cv2.GaussianBlur(grayImg,(21,21),0)

    fgmask=fgbg.apply(img)
    fgmask=cv2.erode(fgmask,None,iterations=2)
    fgmask=cv2.dilate(fgmask,None,iterations=2)



    if first_frame is None:
        first_frame=gussianImg
        continue

    imgdiff = cv2.absdiff(first_frame,gussianImg)
    threshImg = cv2.threshold(imgdiff,25,255,cv2.THRESH_BINARY)[1]
    threshImg=cv2.dilate(threshImg,None,iterations=2)

    if time.time()-last_update > 10:
        first_frame = gussianImg
        last_update=time.time()

    motion_mask = cv2.bitwise_or(threshImg,fgmask)

    cnts,_=cv2.findContours(motion_mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        if cv2.contourArea(c) < area:
            continue
        (x,y,w,h)=cv2.boundingRect(c)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        text="moving object detected"

    print(text)
    cv2.putText(img,text,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)


    cv2.imshow("cameraFeed",img)
    cv2.imshow("Motionmask",motion_mask)
    
    key=cv2.waitKey(1) & 0xff
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()