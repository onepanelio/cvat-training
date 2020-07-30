import pandas as pd
import xml.etree.ElementTree as ET
import csv
import os
from sqlalchemy import create_engine


def str2int(s):
    return int(float(s))


def create_csv(xml_file, gps_csv, video_file, out_path):
   
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
            p[d[0]] = {'timestamp':d[indexes['timestamp']], 'lat':d[indexes['lat']], 'lon':d[indexes['lon']]}


    mylist = []
    for track in root.findall('image'):
        for box in track.findall('box'):
            mydict = box.attrib
            mydict.update(track.attrib)
            mydict.update({'damage':damage_mapping[box.attrib['label']]})
            mydict.update({'video_file':video_file})
            mydict.update(p[track.attrib['id']])

            mylist.append(mydict)
    df = pd.DataFrame(mylist)
    df.xbr = df.xbr.apply(lambda x: str2int(x))
    df.xtl = df.xtl.apply(lambda x: str2int(x))
    df.ybr = df.ybr.apply(lambda x: str2int(x))
    df.ytl = df.ytl.apply(lambda x: str2int(x))
    df.frame = df.id.apply(lambda x: str2int(x))
    df.to_csv(out_path, index=False)

def dump_to_sql(xml_file, gps_csv, video_file):
    user = os.getenv('CRB_SQL_USERNAME')
    password = os.getenv('CRB_SQL_PASSWORD')
    table = os.getenv('CRB_SQL_TABLE')
    print(user)
    csv_file = os.path.join(os.path.basename(video_file)[:-5], '.csv')
    print("Generating CSV file to create SQL database...")
    create_csv(xml_file, gps_csv, video_file, "temp.csv")

    engine = create_engine(f'mysql+pymysql://{user}:{password}@mysql.guaminsects.net/videosurvey')

    df = pd.read_csv("temp.csv")
    print("Making connection request to database...")
    conn = engine.connect()
    conn.execute(f'DROP TABLE IF EXISTS {table};')
    df.to_sql(table, engine, index=False, if_exists="replace")
    conn.execute(f'ALTER TABLE {table} ADD coords POINT;')
    conn.execute(f'UPDATE {table} SET coords=POINT(lon,lat);')
    conn.execute(f'INSERT INTO {table} SELECT * FROM {table};')
    print("Data inserted successfully!")
    conn.close()
