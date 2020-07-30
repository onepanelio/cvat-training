import pandas as pd
import xml.etree.ElementTree as ET
import csv
import os
import sys
from sqlalchemy import create_engine


def str2int(s):
    return int(float(s))


def create_csv(xml_file, gps_csv, video_file, out_path, skip_no, drop_extra):
   
    tree = ET.parse(xml_file)
    root = tree.getroot()
    damage_mapping = {'zero':0, 'light':1, 'medium':2, 'high':3, 'non_recoverable':4}
    
    gps = csv.reader(open(gps_csv).readlines())
    p = {}
    indexes = {'timestamp':1, 'lat':2, 'lon':3}
    for d in gps:
        if 'frame' in d:
            indexes['timestamp'] = d.index('timestamp')
            indexes['lat'] = d.index('lat')
            indexes['lon'] = d.index('lon')
        else:
            if int(d[0]) % skip_no == 0:
                p[d[0]] = {'timestamp':d[indexes['timestamp']], 'lat':d[indexes['lat']], 'lon':d[indexes['lon']]}


    mylist = []
    track_id = 0
    for track in root.findall('image'):
        for box in track.findall('box'):
            mydict = box.attrib
            mydict.update(track.attrib)
            mydict.update({'damage':damage_mapping[box.attrib['label']]})
            mydict.update({'video_file':video_file})
            mydict.update(p[track.attrib['id']])
            # add following two columns just to same structure as objects table
            mydict.update({'keyframe': 1})
            mydict.update({'track_id': track_id})
            track_id += 1
            mylist.append(mydict)

    if mylist:
        # objects are detected
        df = pd.DataFrame(mylist)
        if drop_extra or drop_extra == 'True':
            df.drop(['width', 'height', 'occluded', 'name'], axis=1, inplace=True)
        df = df.rename({'id':'frame'}, axis=1)
        df = df[['frame','keyframe','xtl','ytl','xbr','ybr','track_id','label','timestamp','lat','lon','damage','video_file']]
        df.xbr = df.xbr.apply(lambda x: str2int(x))
        df.xtl = df.xtl.apply(lambda x: str2int(x))
        df.ybr = df.ybr.apply(lambda x: str2int(x))
        df.ytl = df.ytl.apply(lambda x: str2int(x))
        df.frame = df.frame.apply(lambda x: str2int(x))
        df.to_csv(out_path, index=False)
        return True
    
    return False
    
def dump_to_sql(xml_file, gps_csv, video_file, skip_no, write_into_objects, drop_extra, num_frames):
    user = os.getenv('CRB_SQL_USERNAME')
    password = os.getenv('CRB_SQL_PASSWORD')
    table = os.getenv('CRB_SQL_TABLE')
    csv_file = "/mnt/output/"+os.path.basename(video_file)[:-4]+'_skip_{}_numframes_{}.csv'.format(skip_no, num_frames)
    print("Generating CSV file to create SQL database...")
    if not create_csv(xml_file, gps_csv, video_file, csv_file, skip_no, drop_extra):
        sys.exit("Model could not detect any objects. Terminating the SQL dump process...")

    engine = create_engine('mysql+pymysql://{}:{}@mysql.guaminsects.net/videosurvey'.format(user, password))

    try:
        df = pd.read_csv(csv_file)
        print("Making connection request to database...")
        conn = engine.connect()
        conn.execute('DROP TABLE IF EXISTS {};'.format(table))
        df.to_sql(table, engine, index=False, if_exists="replace")
        conn.execute('ALTER TABLE {} ADD coords POINT;'.format(table))
        conn.execute('UPDATE {} SET coords=POINT(lon,lat);'.format(table))
        if write_into_objects or write_into_objects == 'True': #write into objects table
            conn.execute('INSERT INTO objects SELECT * FROM {};'.format(table))
        print("Data inserted successfully!")
        conn.close()
    except RuntimeError as e:
        print("An error occurr while writing to SQL table: ", e)
