import json

with open("data.json") as file:
    data = json.load(file)

# print(data['images'][:5])
c = 0
for r in data['images']:
    # print(r)
    r['id'] = r['id']//7
    text, num = r['file_name'].split("_")
    res = str(int(num)//7).zfill(6)
    r['file_name'] = 'frame_{}'.format(res)
    # print(r)
    # c += 1
    # if c == 10:
    #     break
c = 0
for b in data['annotations']:
    # print(b)
    b['image_id'] = b['image_id']//7
    # print(b)
    b['segmentation'] = {}
    c += 1
    # if c ==34:

    #     break

with open('custom2.json', 'w') as f:  # writing JSON object
    json.dump(data, f)
 