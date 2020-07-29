import cv2
import os
import sys

def convert_to_frames(video_path, out_path, skip_no=7):
    cap = cv2.VideoCapture(video_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    frame_no = 0
    while True:
        # print("frame no", frame_no)
        ret, frame = cap.read()
        if ret:
            if frame_no % int(skip_no) == 0:
                cv2.imwrite(os.path.join(out_path, "{}.jpg".format(frame_no)), frame)
            frame_no += 1
        else:
            break

if __name__ == "__main__":
    # sameple: python3 video_to_frames.py samepl.mp4 output_folder/ 7
    print(sys.argv[1], sys.argv[2])
    convert_to_frames(sys.argv[1], sys.argv[2], sys.argv[3])