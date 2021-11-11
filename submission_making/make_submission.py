from glob import glob
import json
import imageio
import numpy as np
from collections import OrderedDict

target_folder = 'submission_image'
output_file = 'submit.json'

set5_json = OrderedDict()

for i,img_path in enumerate(glob(target_folder+'/*')):
    img = imageio.imread(img_path)
    set5_json[str(i)] = img.tolist()

# print(json.dumps(set5_json, ensure_ascii=False, indent="\t"))

with open(output_file, 'w', encoding="utf-8") as make_file:
    json.dump(set5_json, make_file, ensure_ascii=False, indent='\t')

print("{}...saved!!".format(output_file))




