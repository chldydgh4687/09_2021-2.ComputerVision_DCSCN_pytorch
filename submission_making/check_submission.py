from glob import glob
import json
import imageio
import numpy as np
from collections import OrderedDict

main_file = "bicubic.json"
check_file = "submit.json"

with open(main_file, "r", encoding="utf-8") as make_file:
    org_ = json.load(make_file)

with open(check_file, 'r', encoding="utf-8") as make_file:
    json_data = json.load(make_file)

print("bicubic_shape {} submit_shape {}".format(org_.keys(),json_data.keys()))
print("bicubic_ {} submit_ {}".format(np.array(org_["0"]).shape, np.array(json_data["0"]).shape))
print("bicubic_ {} submit_ {}".format(np.array(org_["1"]).shape, np.array(json_data["1"]).shape))
print("bicubic_ {} submit_ {}".format(np.array(org_["2"]).shape, np.array(json_data["2"]).shape))
print("bicubic_ {} submit_ {}".format(np.array(org_["3"]).shape, np.array(json_data["3"]).shape))
print("bicubic_ {} submit_ {}".format(np.array(org_["4"]).shape, np.array(json_data["4"]).shape))
