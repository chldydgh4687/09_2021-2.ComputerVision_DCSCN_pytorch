from glob import glob
import json
import imageio
import os
import numpy as np
from collections import OrderedDict

#########################################

output_file = 'submit.json'

#########################################

target_folder = 'submission_image'
main_file = "Fixed_Submit/bicubic.json"
set5_json = OrderedDict()

for i,img_path in enumerate(glob(target_folder+'/*')):
    img = imageio.imread(img_path)
    set5_json[str(i)] = img.tolist()

# print(json.dumps(set5_json, ensure_ascii=False, indent="\t"))

with open(output_file, 'w', encoding="utf-8") as make_file:
    json.dump(set5_json, make_file, ensure_ascii=False, indent='\t')


############check_submission#################

print("[Submission CrossChecking..]")

with open(main_file, "r", encoding="utf-8") as make_file:
    org_ = json.load(make_file)

with open(output_file, 'r', encoding="utf-8") as make_file:
    json_data = json.load(make_file)

submit_key = list(json_data)
try:
    for i,key in enumerate(org_.keys()):
        if key == submit_key[i]:
            print("bicubic_key {} submit_key {} it have!!".format(key,submit_key[i]))
        else:
            print("ERROR : Check the number of image in submission_image folder!( must be 5 )")
            os.remove(output_file)
            exit()
except:
    print("ERROR : Check the number of image in submission_image folder!( must be 5 )")
    os.remove(output_file)
    exit()

if np.array(org_["0"]).shape == np.array(json_data["0"]).shape:
    print("bicubic_ {} submit_ {} same!!".format(np.array(org_["0"]).shape, np.array(json_data["0"]).shape))
else:
    print("ERROR : bicubic_ {} submit_ {} different size..!".format(np.array(org_["0"]).shape, np.array(json_data["0"]).shape))
    print("check images order between submission_image and SET5 folder!")
    os.remove(output_file)
    print("{}...deleted!!".format(output_file))
    exit()

if np.array(org_["1"]).shape == np.array(json_data["1"]).shape:
    print("bicubic_ {} submit_ {} same!!".format(np.array(org_["1"]).shape, np.array(json_data["0"]).shape))
else:
    print("ERROR : bicubic_ {} submit_ {} different size..!".format(np.array(org_["1"]).shape, np.array(json_data["1"]).shape))
    print("check images order between submission_image and SET5 folder!")
    os.remove(output_file)
    print("{}...deleted!!".format(output_file))
    exit()

if np.array(org_["2"]).shape == np.array(json_data["2"]).shape:
    print("bicubic_ {} submit_ {} same!!".format(np.array(org_["2"]).shape, np.array(json_data["2"]).shape))
else:
    print("ERROR : bicubic_ {} submit_ {} different size..!".format(np.array(org_["2"]).shape, np.array(json_data["2"]).shape))
    print("check images order between submission_image and SET5 folder!")
    os.remove(output_file)
    print("{}...deleted!!".format(output_file))
    exit()

if np.array(org_["3"]).shape == np.array(json_data["3"]).shape:
    print("bicubic_ {} submit_ {} same!!".format(np.array(org_["3"]).shape, np.array(json_data["3"]).shape))
else:
    print("ERROR : bicubic_ {} submit_ {} different size..!".format(np.array(org_["3"]).shape, np.array(json_data["3"]).shape))
    print("check images order between submission_image and SET5 folder!")
    os.remove(output_file)
    print("{}...deleted!!".format(output_file))
    exit()

if np.array(org_["4"]).shape == np.array(json_data["4"]).shape:
    print("bicubic_ {} submit_ {} same!!".format(np.array(org_["4"]).shape, np.array(json_data["4"]).shape))
else:
    print("ERROR : bicubic_ {} submit_ {} different size..!".format(np.array(org_["4"]).shape, np.array(json_data["4"]).shape))
    print("check images order between submission_image and SET5 folder!")
    os.remove(output_file)
    print("{}...deleted!!".format(output_file))
    exit()
print("[Submission CrossChecking..]...done!!")
print("{}...saved!!".format(output_file))


