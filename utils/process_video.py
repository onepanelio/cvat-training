import cv2
import math
import argparse
import os

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
            ret, frame = self.cap.read()
            if ret:
                if frame_no % skip_no==0:
                    out.write(frame)
                frame_no += 1
            else:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_video", help="Model Pipeline Path")
    parser.add_argument("--output_path", help="Input Model Path")
    parser.add_argument("--skip", default=7, type=int, help="label_path")
 
    args = parser.parse_args()
    print("Working dir: {}".format(os.getcwd()))
    videos = args.input_video.split(",")
    for video in videos:
        v = VideoEditor(video)
        basename = os.path.basename(video)
        extension = basename[-4:]
        print("Storing {} in /mnt/output...".format(basename[:-4]+'_processed'+extension))
        
        v.skip_frame_write(args.skip, os.path.join("/mnt/output/", basename[:-4]+'_processed'+extension))
        os.chdir("/mnt/output/")
        os.listdir()