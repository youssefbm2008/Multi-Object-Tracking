#haarCascade Face and Eye detector
import numpy as np
import cv2
import sys
import time
import pandas as pd
import numpy as np



#function to check if a face region in the frame is being tracked
def if_face_tracked(x_bar, y_bar, t_x, t_y, t_w, t_h):
        #calculate the centerpoint
        x_bar = x + 0.5 * w
        y_bar = y + 0.5 * h
        #calculate the centerpoint
        t_x_bar = t_x + 0.5 * t_w
        t_y_bar = t_y + 0.5 * t_h
        
        #check if the centerpoint of the face is within the 
        #rectangleof a tracker region. Also, the centerpoint
        #of the tracker region must be within the region 
        #detected as a face. If both of these conditions hold
        #we have a match
        if ( ( t_x <= x_bar   <= (t_x + t_w)) and 
             ( t_y <= y_bar   <= (t_y + t_h)) and 
             ( x   <= t_x_bar <= (x   + w  )) and 
             ( y   <= t_y_bar <= (y   + h  ))):
            return True
        else:
            return False





face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_eye.xml')
#cap = cv2.VideoCapture(0)
#cap.release()
cap=cv2.VideoCapture(0)
cv2.destroyAllWindows()

#initial detection of faces
while(1):
    ret,frame = cap.read()
    frame = np.array(frame)
    #print (type(frame))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = np.array(face_cascade.detectMultiScale(gray, 1.3, 5))
    cv2.imshow('output',frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        cap.release()
        cv2.destroyAllWindows()
        sys.exit()


    #x y w h for 1 face and face1 image
    #facem stores the multiple face frames
    facem=[]
    if len(faces)!=0:
        faces_x=faces[:,0]
        faces_y=faces[:,1]
        faces_w=faces[:,2]
        faces_h=faces[:,3]
        #print ("faces_x", faces_x)
        #print ("faces_w", faces_w)
        #print ("faces_x+faces_w", faces_x+faces_w)
        #face1 is the image of extracted face
        for x,y,w,h in zip(faces_x,faces_y,faces_w,faces_h):
            face1=frame[y:y+h, x:x+w]
            facem.append(face1)
            cv2.imshow('face',face1)
        #print "faces", faces
        #print "face1:", face1
        #print ("facem len:", len(facem) )
        #if len(facem)>1:
        cv2.imshow('face1', facem[0])
        #print ("face1:dtype" , type(facem[0]))
        #print ("face1", facem[0])
        break



columns = ['trackerID','timer']
timer_all = pd.DataFrame(columns=columns)






#kcf tracker initialisation
trackerm=[]
okm=[]
for x,y,w,h in zip(faces_x,faces_y,faces_w,faces_h):
    tracker = cv2.TrackerKCF_create()
    bbox =  (x,y, w, h) 
    ok = tracker.init(frame, bbox)
    trackerm.append(tracker)
    okm.append(ok)
    timer_all.loc[len(timer_all)] = [tracker,time.time()]
    print(type(tracker))
    print(timer_all)

#print "ok size:", len(ok)
#sys.exit()

fps=0
fps_counter=0
#timer=time.time()

frames=0
people_count=len(faces)

while True:
    # Read a new frame
    ok, frame = cap.read()
    if not ok:
        break
    #looking for new faces in each frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = np.array(face_cascade.detectMultiScale(gray, 1.3, 5))
    
    # Update tracker
    #ok, bbox = tracker.update(frame)
    okm=[]
    bboxm=[]
    
    for tracker in trackerm:
        ok, bbox = tracker.update(frame)
        #print(ok, 'tracker', tracker)
        okm.append(ok)
        bboxm.append(bbox)
        if ok == False:
                trackerm.remove(tracker)
                #print('tracker removed')
                timer = timer_all[timer_all['trackerID']==tracker]['timer'].iat[0]
                print('Person Tracking time', time.time()-timer,'s')
                
        
    if len(faces)>len(trackerm):
            print('new faces in frame')
            
            for (_x,_y,_w,_h) in faces:
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)
                
                #calculate the centerpoint
                x_bar = x + 0.5 * w		
                y_bar = y + 0.5 * h
                if len(bboxm)>0:
                        for (t_x,t_y,t_w,t_h) in bboxm:	
                                t_x = int(t_x)	
                                t_y = int(t_y)	
                                t_w = int(t_w) 	     		
                                t_h = int(t_h)	
                                if if_face_tracked(x_bar, y_bar, t_x, t_y, t_w, t_h):	
                                        #pass if face is already tracked
                                        print('Face detected and tracked')
                                        #print('step', x,y,w,h)
                                        break
                                else: 	
                                        # Face found in Faces but not tracked
                                        print('new tracker added')
                                        #print('step', x,y,w,h)
                                        bbox =  (_x, _y, _w, _h) 
                                        tracker = cv2.TrackerKCF_create()
                                        ok = tracker.init(frame, bbox)
                                        trackerm.append(tracker)
                                        okm.append(ok)
                                        people_count +=1
                                        timer_all.loc[len(timer_all)] = [tracker,time.time()]
                                        break
                else:
                        print('bbox empty, initializing')
                        bbox =  (_x, _y, _w, _h) 
                        tracker = cv2.TrackerKCF_create()
                        ok = tracker.init(frame, bbox)
                        trackerm.append(tracker)
                        okm.append(ok)
                        people_count +=1
                        timer_all.loc[len(timer_all)] = [tracker,time.time()]
                        break
        # break if 'q' key pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
                        
                        
                            
        

    # Draw bounding box
#    okm=np.array(okm)
#    bboxm=np.array(bboxm)
#    for box_x, box_y, box_w, box_h in zip(bboxm[okm==True][:,0], bboxm[okm==True][:,1],bboxm[okm==True][:,2],bboxm[okm==True][:,3]): #try boxm[ok==True][0] for box_x .... 
#        p1 = (int(box_x), int(box_y))
#        p2 = (int(box_x + box_w), int(box_y + box_h))
#        cv2.rectangle(frame, p1, p2, (0,255,0), 2, 1)
    

        
        


        
    
    #timer,fps, refind:finds the face and features again
#    fps_counter=fps_counter+1
#    if(time.time()-timer>1):
#        print (timer,":",fps_counter)
#        fps=fps_counter
#        fps_counter=0
#        timer=time.time()
#        refind_timer=refind_timer+1
#        #removing is ok cond gives good fit bounding box but reduces fps , significant if box is large 
#        #if ok==0:
#        if refind_timer==2:
#            print ("before refine bbox:", bboxm)
#            refind()
#            print ("after refine bbox:", bboxm)
#            refind_timer=0
#        #tracker = cv2.TrackerKCF_create()
#        #ok = tracker.init(frame, bbox)
#        #print "refined ok", ok
#
#    
#
#    #frames, refind:finds the face and features again
#    frames=frames+1
#    if(frames>fps):
#        frames=0
#        #refind()
#    cv2.putText(frame, "Fps:"+str(fps), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255,0,0),2)
                
    #output
    cv2.imshow('output', frame)



# In[ ]:




# In[17]:

#Destroy any OpenCV windows and exit the application
cap.release()
cv2.destroyAllWindows()
print('---------- Final Tracking Stats ------------')
print('Total engaged audience',people_count)
