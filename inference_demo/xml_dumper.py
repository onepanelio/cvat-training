from xml.sax.saxutils import XMLGenerator
from collections import OrderedDict

class XmlAnnotationWriter:
    def __init__(self, file):
        self.version = "1.1"
        self.file = file
        self.xmlgen = XMLGenerator(self.file, 'utf-8')
        self._level = 0

    def _indent(self, newline = True):
        if newline:
            self.xmlgen.ignorableWhitespace("\n")
        self.xmlgen.ignorableWhitespace("  " * self._level)

    def _add_version(self):
        self._indent()
        self.xmlgen.startElement("version", {})
        self.xmlgen.characters(self.version)
        self.xmlgen.endElement("version")

    def open_root(self):
        self.xmlgen.startDocument()
        self.xmlgen.startElement("annotations", {})
        self._level += 1
        self._add_version()

    def _add_meta(self, meta):
        self._level += 1
        for k, v in meta.items():
            if isinstance(v, OrderedDict):
                self._indent()
                self.xmlgen.startElement(k, {})
                self._add_meta(v)
                self._indent()
                self.xmlgen.endElement(k)
            elif isinstance(v, list):
                self._indent()
                self.xmlgen.startElement(k, {})
                for tup in v:
                    self._add_meta(OrderedDict([tup]))
                self._indent()
                self.xmlgen.endElement(k)
            else:
                self._indent()
                self.xmlgen.startElement(k, {})
                self.xmlgen.characters(v)
                self.xmlgen.endElement(k)
        self._level -= 1

    def add_meta(self, meta):
        self._indent()
        self.xmlgen.startElement("meta", {})
        self._add_meta(meta)
        self._indent()
        self.xmlgen.endElement("meta")

    def open_track(self, track):
        self._indent()
        self.xmlgen.startElement("track", track)
        self._level += 1

    def open_image(self, image):
        self._indent()
        self.xmlgen.startElement("image", image)
        self._level += 1

    def open_box(self, box):
        self._indent()
        self.xmlgen.startElement("box", box)
        self._level += 1

    def open_polygon(self, polygon):
        self._indent()
        self.xmlgen.startElement("polygon", polygon)
        self._level += 1

    def open_polyline(self, polyline):
        self._indent()
        self.xmlgen.startElement("polyline", polyline)
        self._level += 1

    def open_points(self, points):
        self._indent()
        self.xmlgen.startElement("points", points)
        self._level += 1

    def open_cuboid(self, cuboid):
        self._indent()
        self.xmlgen.startElement("cuboid", cuboid)
        self._level += 1

    def open_tag(self, tag):
        self._indent()
        self.xmlgen.startElement("tag", tag)
        self._level += 1

    def add_attribute(self, attribute):
        self._indent()
        self.xmlgen.startElement("attribute", {"name": attribute["name"]})
        self.xmlgen.characters(attribute["value"])
        self.xmlgen.endElement("attribute")

    def close_box(self):
        self._level -= 1
        self._indent()
        self.xmlgen.endElement("box")

    def close_polygon(self):
        self._level -= 1
        self._indent()
        self.xmlgen.endElement("polygon")

    def close_polyline(self):
        self._level -= 1
        self._indent()
        self.xmlgen.endElement("polyline")

    def close_points(self):
        self._level -= 1
        self._indent()
        self.xmlgen.endElement("points")

    def close_cuboid(self):
        self._level -= 1
        self._indent()
        self.xmlgen.endElement("cuboid")

    def close_tag(self):
        self._level -= 1
        self._indent()
        self.xmlgen.endElement("tag")

    def close_image(self):
        self._level -= 1
        self._indent()
        self.xmlgen.endElement("image")

    def close_track(self):
        self._level -= 1
        self._indent()
        self.xmlgen.endElement("track")

    def close_root(self):
        self._level -= 1
        self._indent()
        self.xmlgen.endElement("annotations")
        self.xmlgen.endDocument()

def pairwise(iterable):
    a = iter(iterable)
    return zip(a, a)

def dump_as_cvat_annotation(file_object, annotations):
    from collections import OrderedDict
    dumper = XmlAnnotationWriter(file_object)
    dumper.open_root()
    dumper.add_meta(annotations['meta'])

    for frame_annotation in annotations['frames']:
        frame_id = frame_annotation['frame']
        dumper.open_image(OrderedDict([
            ("id", str(frame_id)),
            ("name", 'frame_'+str(frame_id).zfill(6)),
            ("width", str(frame_annotation['width'])),
            ("height", str(frame_annotation['height']))
        ]))

        for shape in frame_annotation['shapes']:
            dump_data = OrderedDict([
                ("label", shape['label']),
                ("occluded", str(int(shape['occluded']))),
            ])

            if shape['type'] == "rectangle":
                dump_data.update(OrderedDict([
                    ("xtl", "{:.2f}".format(shape['points'][0])),
                    ("ytl", "{:.2f}".format(shape['points'][1])),
                    ("xbr", "{:.2f}".format(shape['points'][2])),
                    ("ybr", "{:.2f}".format(shape['points'][3]))
                ]))
        
            else:
                dump_data.update(OrderedDict([
                    ("points", ';'.join((
                        ','.join((
                            "{:.2f}".format(x),
                            "{:.2f}".format(y)
                        )) for x, y in pairwise(shape['points']))
                    )),
                ]))


            if shape['type'] == "rectangle":
                dumper.open_box(dump_data)
            elif shape['type'] == "polygon":
                dumper.open_polygon(dump_data)

         

            if shape['type'] == "rectangle":
                dumper.close_box()
            elif shape['type'] == "polygon":
                dumper.close_polygon()

        dumper.close_image()
    dumper.close_root()


if __name__ =="__main__":
    annotations = {'meta':{'task': OrderedDict([('id','48'),('name','test'),('size','44'),('mode','interpolation'),('z_order',False),('labels', [('label',[('name','cut')]),('label', [('name','zero')])])])},'frames':[{'frame':0,'width':2280,'height':1920, 'shapes':[{'type':'polygon','label':'cut','occluded':0,'points':[10,5,11,6]}]}]}
    dump_as_cvat_annotation(open("cvat_anno_test.xml","w"), annotations)