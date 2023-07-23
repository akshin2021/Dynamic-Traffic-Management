import numpy as np
import cv2
from threading import Thread
def process(origFrame, prev):
    try:
        frame = cv2.cvtColor(origFrame.copy(), cv2.COLOR_BGR2GRAY)
        roi = frame[200: 620, 500: 1000]
        backgroundObject = cv2.createBackgroundSubtractorMOG2(detectShadows = True)
        thresh= backgroundObject.apply(roi)
        if prev.shape!=(0, 0, 0): 
                diff = cv2.absdiff(prev, roi)
                ret, thresh = cv2.threshold(diff, 75, 255, cv2.THRESH_BINARY)
                kernel = np.ones((5,5),np.uint8)
                dilated = cv2.dilate(thresh,kernel,iterations = 1)
                blurred = cv2.blur(dilated, (5, 5))
                contours, hierarchy = cv2.findContours(blurred.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

                contours = [contour for contour in contours if cv2.contourArea(contour)>100]
                contoursLen = str(len(contours))

                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(origFrame, (x+500, y+200), (x+w+500, y+h+200), (0, 255, 0), 2)
                    cv2.putText(origFrame, contoursLen, (1200, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        prev = frame[200: 620, 500: 1000]
    except Exception as e:
        pass
    return origFrame, prev


from threading import Thread, Lock
import cv2

class WebcamVideoStream :
    def __init__(self, src = 0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()
        self.prev = np.empty([0,0,0])

    def start(self) :
        if self.started :
            print( "already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self) :
        while self.started :
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.frame, self.prev = process(self.frame, self.prev)
            try:
                self.frame = cv2.resize(self.frame, (720, 400))
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
        return frame

    def stop(self) :
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()

if __name__ == "__main__" :
    vs = WebcamVideoStream(src = "highway1.mp4").start()
    vs2 = WebcamVideoStream(src = "highway1.mp4").start()
    vs3 = WebcamVideoStream(src = "highway1.mp4").start()
    vs4 = WebcamVideoStream(src = "highway1.mp4").start()
                  
    c = 0

    while True :
        try:
            c+=1
            frame = vs.read()
            # frame, prev1 = process(frame, prev1)
            frame2 = vs2.read()
            # frame2, prev2 = process(frame2, prev2)
            frame3 = vs3.read()
            # frame3, prev3 = process(frame3, prev3)
            frame4 = vs4.read()
            # frame4, prev4 = process(frame4, prev4)

            # frame = cv2.resize(frame, (720, 400))
            # frame2 = cv2.resize(frame2, (720, 400))
            # frame3 = cv2.resize(frame3, (720, 400))
            # frame4 = cv2.resize(frame4, (720, 400))

            if np.array(frame).shape == np.array(frame2).shape : 
                hori1 = np.concatenate((frame, frame2), axis = 1)
                hori2 = np.concatenate((frame3, frame4), axis = 1)
                vert = np.concatenate((hori1, hori2), axis = 0)
                cv2.imshow("all4", vert)
        except Exception as e:
            vs = WebcamVideoStream(src = "highway1.mp4").start()
            vs2 = WebcamVideoStream(src = "highway1.mp4").start()
            vs3 = WebcamVideoStream(src = "highway1.mp4").start()
            vs4 = WebcamVideoStream(src = "highway1.mp4").start()

        if cv2.waitKey(1) & 0xFF == ord('q') :
            break
    vs.stop()
    vs2.stop()
    vs3.stop()
    vs4.stop()
    cv2.destroyAllWindows()