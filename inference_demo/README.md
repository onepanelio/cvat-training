## Running Inference using Multiple Models

`demo.py` runs inference on given input video and outputs video with bounding boxes/polygons drawn.

Install prerequisites.
```python
pip3 install -r requirements.txt
```

```python
python3 demo.py --type=both \ 
    --video=demo.mp4 \                         #path to input video
    --gps_csv=demo.csv \                       #path to csv file containing gps data
    --od_model=frozen_inference_graph.pb \     #path to frozen graph for object detection
    --mask_model=mask_rcnn_cvat.h5 \           #path to maskrcnn model
    --output_video=output.mp4  \               #path to output video
    --survey_type=v_shape                      #what should be included in the outputted geojson file
```
