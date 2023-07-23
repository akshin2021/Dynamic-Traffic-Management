import random
import time
import numpy as np
import threading
from threading import Thread, Lock
import cv2

from threading import Thread, Lock
import cv2

class WebcamVideoStream :
     def __init__(self, src = 0):
          self.stream = cv2.VideoCapture(src)
          (self.grabbed, self.frame) = self.stream.read()
          self.started = False
          self.read_lock = Lock()
          self.traffic = random.random()

     def start(self) :
          if self.started :
               print( "already started!!")
               return None
          self.started = True
          self.thread = Thread(target=self.update, args=())
          self.thread.start()
          return self

     def update(self) :
          while True :
               if self.started:
                    (grabbed, frame) = self.stream.read()
                    self.read_lock.acquire()
                    self.grabbed, self.frame = grabbed, frame
                    try:
                         self.frame = cv2.resize(self.frame, (720, 400))
                         self.traffic = random.random()
                    except Exception as e:
                         pass
                    self.read_lock.release()

     def read(self) :
          self.read_lock.acquire()
          try:
               frame = self.frame.copy()
          except Exception as e:
               frame = None
          self.read_lock.release()
          return frame, self.traffic

     def stop(self) :
          self.started = False
          self.thread.join()

     def __exit__(self, exc_type, exc_value, traceback) :
          self.stream.release()


if __name__ == "__main__" :
     vs = WebcamVideoStream(src = "highway.mp4").start()
     vs2 = WebcamVideoStream(src = "highway.mp4").start()
     vs3 = WebcamVideoStream(src = "highway.mp4").start()
     vs4 = WebcamVideoStream(src = "highway.mp4").start()

     arr = [0, 0, 0, 0]
     road = 0
     timer = [15]
     reduced = [False]

     
     def reduce():
          arr[road] = max(0, arr[road]-5)
          timer[0] -= 1
          print(timer[0], road)
          reduced[0] = False

     frame, traffic1 = vs.read()
     frame2, traffic2 = vs2.read()
     frame3, traffic3 = vs3.read()
     frame4, traffic4 = vs4.read()

     initial = frame
     road = 0

     while True :
          try:
               
               if timer[0] == 0:
                    road+=1
                    road%=4
                    timer[0] = 30 if arr[road]>50 else 20 if 20<arr[road]<=50 else 15
               frame, traffic1 = vs.read()
               frame2, traffic2 = vs2.read()
               frame3, traffic3 = vs3.read()
               frame4, traffic4 = vs4.read()
               arr[0]+=traffic1*0.06
               arr[1]+=traffic2*0.05
               arr[2]+=traffic3*0.03
               arr[3]+=traffic4*0.06
               if not reduced[0]:
                    threading.Timer(1, reduce).start()
                    reduced[0] = True

               if np.array(frame).shape == np.array(frame2).shape :
                    x1 = frame if road==0 else initial.copy()
                    x2 = frame2 if road==1 else initial.copy()
                    x3 = frame3 if road==2 else initial.copy()
                    x4 = frame4 if road==3 else initial.copy()
                    cv2.putText(x1, str(int(arr[0])), (600, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.circle(img=x1, center = (600,260), radius =36, color = (0,255,0) if road==0 else (0, 0, 255), thickness=-1)
                    cv2.putText(x2, str(int(arr[1])), (60, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.circle(img=x2, center = (60,260), radius =36, color = (0,255,0) if road==1 else (0, 0, 255), thickness=-1)
                    cv2.putText(x3, str(int(arr[2])), (600, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.circle(img=x3, center = (600,120), radius =36, color = (0,255,0) if road==2 else (0, 0, 255), thickness=-1)
                    cv2.putText(x4, str(int(arr[3])), (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.circle(img=x4, center = (60,120), radius =36, color = (0,255,0) if road==3 else (0, 0, 255), thickness=-1)
                    hori1 = np.concatenate((x1, x2), axis = 1)
                    hori2 = np.concatenate((x3, x4), axis = 1)
                    vert = np.concatenate((hori1, hori2), axis = 0)
                    cv2.putText(vert, str(timer[0]), (700, 410), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow("all4", vert)

          except Exception as e:
               try:
                    frame.any()
               except Exception as e2:
                    vs = WebcamVideoStream(src = "highway1.mp4").start()
               try:
                    frame2.any()
               except Exception as e2:
                    vs2 = WebcamVideoStream(src = "highway1.mp4").start()
               try:
                    frame3.any()
               except Exception as e2:
                    vs3 = WebcamVideoStream(src = "highway1.mp4").start()
               try:
                    frame4.any()
               except Exception as e2:
                    vs4 = WebcamVideoStream(src = "highway1.mp4").start()

          if cv2.waitKey(1) & 0xFF == ord('q') :
               break

     vs.stop()
     vs2.stop()
     vs3.stop()
     vs4.stop()
     cv2.destroyAllWindows()