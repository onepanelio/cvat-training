import cv2
import math

class VideoEditor:
    def __init__(self, path):
        self.path = path    
        self.cap = cv2.VideoCapture(self.path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = math.ceil(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def __repr__(self):
        return "File: " + self.path + "\n" + \
                "Width: "+ str(self.width) + "\n" + \
                "Height: "+ str(self.height) + "\n" + \
                "FPS: "+ str(self.fps) + '\n' +\
                "Frame Count: "+str(self.frame_count)
    
    def skip_frame_write(self, skip_no, output_path):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width,self.height))
        frame_no = 0
        while True:
            print(frame_no)
            ret, frame = self.cap.read()
            if ret:
                if frame_no % skip_no==0:
                    print("writing frame ", frame_no)
                    out.write(frame)
                frame_no += 1
            else:
                break

v = VideoEditor("20200627_135552.mp4")
print(v)
v.skip_frame_write(7, "20200627_135552_processed.mp4")