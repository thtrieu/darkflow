from darkflow.net.build import TFNet
import cv2

options = {"model": "cfg/yolo-voc-2c.cfg", "load": 25375, "threshold": 0.0, 'gpu':1.0} # Change this line as per your needs

tfnet = TFNet(options)
PATH = '' # Insert video path to see real time results on a video or 0 to test it real time

cap = cv2.VideoCapture(PATH)
while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        h, w, d = frame.shape

        result = tfnet.return_predict(frame)
        
        for a in result:
            label = a['label']
            tl = a['topleft']
            x1 = tl['x']
            y1 = tl['y']
            br = a['bottomright']
            x2 = br['x']
            y2 = br['y']
            color_box = (0, 255, 0) # You can change color according to classes as well
            color_label = (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_box, 4)
            cv2.putText(frame, label, (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_label, thickness=2, lineType=2)


        
        
        cv2.imshow('frame', frame)	
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        break            

cap.release()
cv2.destroyAllWindows()        