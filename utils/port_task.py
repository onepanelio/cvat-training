import os
import argparse
import fnmatch
import shutil
import json

def make_image_list(path_to_data):
	def get_image_key(item):
		return int(os.path.splitext(os.path.basename(item))[0])

	image_list = []
	for root, _, filenames in os.walk(path_to_data):
		for filename in fnmatch.filter(filenames, '*.jpg'):
				image_list.append(os.path.join(root, filename))

	image_list.sort(key=get_image_key)
	return image_list


def find_image_name(annotation):
	with open(annotation) as f:
		data = json.load(f)
	if len(data['images']) <1:
		raise RuntimeError("There are no annotated images.")
	if "input/datasets" in data['images'][0]['file_name']:
		raise RuntimeError("The images used in this task were attached from a Onepanel dataset. So you don't need to rename images. Just attach the same dataset to new workspace and create a new task with those images.")
	mapping = {}
	for i in data['images']:
		mapping[i['id']] = i['file_name']
	return mapping


def main(dir, annotation):

	image_list = make_image_list(dir)
	if not os.path.exists(os.path.join(dir, "single")):
		os.mkdir(os.path.join(dir, "single"))
	mapping = find_image_name(annotation)
	for img in image_list:
		new_name = mapping[int(os.path.basename(img).split(".")[0])]
		shutil.copy(img, os.path.join(dir, "single",new_name))
		break


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dir", help="path to image dir")
	parser.add_argument("--annotation",help="path to dumped json annotation")
	args = parser.parse_args()
	main(args.dir, args.annotation)