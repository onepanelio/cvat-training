import pandas as pd
from geojson import Point, Feature, FeatureCollection, dump, LineString
import ffmpeg
from datetime import datetime, timedelta

class GPSLogger:
    def __init__(self, video_path, gps_path):
        self.video = video_path
        self.gps_csv = gps_path
        self.features = []
        self._prepare()
    
    def _prepare(self):
        self.gps_data = pd.read_csv(self.gps_csv, parse_dates=['time'])
        self.gps_data['time'] = self.gps_data['time'].dt.tz_localize(None)

    def get_start_stop(self):
        metadata = ffmpeg.probe(self.video)
        ct = metadata['streams'][0]['tags']['creation_time']
        ct = datetime.strptime(ct.replace('T',' ').replace('-',':').split('.')[0], '%Y:%m:%d %H:%M:%S')    
        start = ct - timedelta(seconds=float(metadata['streams'][0]['duration']))
        return start, ct
    
    def update_features(self, frame_no, others):
        # others could be anything based on the requirements
        start, stop = self.get_start_stop()
        df = self.gps_data[(self.gps_data.time>=start) & (self.gps_data.time<=stop)]
        linestring = LineString(list(zip(df.lon,df.lat)))
        linecolor = '#00ff00'
        feature = Feature(geometry=linestring, 
                        properties={"name": (self.video.split('/')[-1]).split('.')[0], "stroke": linecolor})
        self.features.append(feature)

    def dump_geojson(self, output_path=None):
        if output_path is None:
            output_path = self.video.replace('.mp4','.geojson')
        features = FeatureCollection(self.features)
        with open(output_path, 'w') as f:
            dump(features, f)
            print(f"Dumped {output_path}.")