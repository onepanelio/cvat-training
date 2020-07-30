import cv2
import math
import argparse
import csv
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
    
    def skip_frame_write(self, skip_no, output_path, csv_path, num_frames):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width,self.height))
        frame_no = 0
        while True:
            # print("frame no", frame_no)
            ret, frame = self.cap.read()
            if ret:
                if frame_no % skip_no==0:
                    # print("writing frame", frame_no)
                    out.write(frame)
                if (frame_no // skip_no) + 1 == num_frames:
                    break
                frame_no += 1
            else:
                break
        inp = open("/mnt/data/datasets/gps.csv", 'r')
        out = open("/mnt/output/"+str(num_frames)+'_'+os.path.basename(csv_path), 'w')
        writer = csv.writer(out)
        for row in csv.reader(inp):
            if row[0] == "frame" or int(row[0]) % skip_no == 0:
                writer.writerow(row)
            if row[0] != "frame" and (int(row[0]) // skip_no) + 1 == num_frames:
                break
        inp.close()
        out.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--skip", default=7, type=int, help="label_path")
    parser.add_argument("--video", help="name of video file")
    parser.add_argument("--csv_file", help="path to gps-csv file")
    parser.add_argument("--num_frames", default=None, help="number of frames to write")
 
    args = parser.parse_args()
    print("Working dir: {}".format(os.getcwd()))
    video = "/mnt/data/datasets/temp.mp4"
    print("Processing video..", video)
    v = VideoEditor(video)
    basename = os.path.basename(args.video)
    extension = basename[-4:]
    print("Storing {} in /mnt/output...".format(basename[:-4]+'_processed'+extension))
    #set default for num_frames to total frames
    if args.num_frames == 'None':
        args.num_frames = v.frame_count
    print("Output video will have {} frames".format(args.num_frames))
    v.skip_frame_write(args.skip, os.path.join("/mnt/output/", basename[:-4]+'_processed_'+str(args.num_frames)+extension), args.csv_file, int(args.num_frames))
    