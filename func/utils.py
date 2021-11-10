import numpy as np
import os
from PIL import Image
import imageio
from glob import glob
import torch

def SR_FOLDER_GENERATE():
    aug_output = 'augmented_data/train_org/'
    sr_data = "augmented_data/train_sr"
    hr = "augmented_data/train_sr/HR"
    lrbi = "augmented_data/train_sr/LRBICUBIC"
    lrx2 = "augmented_data/train_sr/LRX2"

    try:
        os.mkdir(aug_output)
    except:
        pass
    try:
        os.mkdir(sr_data)
    except:
        pass
    try:
        os.mkdir(hr)
    except:
        pass
    try:
        os.mkdir(lrbi)
    except:
        pass
    try:
        os.mkdir(lrx2)
    except:
        pass

def progressing_bar(count,length):
    if (float(count) / float(length) * 100) % 10.0 == 0:
        print("split progressing : {} {}/{}".format(round(float(count) / float(length) * 100),count,length))

def build_data(train_list,batch_size, scale, BICUBIC_DIR, LRX2_DIR, HR_DIR):

    save_flag = 0

    stride = batch_size // scale
    hr_stride = batch_size
    hr_size = batch_size * scale

    hr_batch_number = 0
    lr_batch_number = 0
    bi_batch_number = 0
    count = 0

    hr_list = []
    lr_list = []
    bi_list = []

    for image_path in train_list:
        gt_img = imageio.imread(image_path)
        gt_img = set_image_alignment(gt_img, 2)

        gt_img_y = convert_rgb_to_y(gt_img)
        lr_img_y = resize_image_by_pil(gt_img_y, 0.5)
        bi_img_y = resize_image_by_pil(lr_img_y, scale)

        gt_y_data = get_split_images(gt_img_y, hr_size, stride=hr_stride)
        lr_y_data = get_split_images(lr_img_y, batch_size, stride=stride)
        bi_y_data = get_split_images(bi_img_y, hr_size, stride=hr_stride)

        hr_list.append(gt_y_data)
        lr_list.append(lr_y_data)
        bi_list.append(bi_y_data)

        count+=1
        progressing_bar(count,len(train_list))
        # [OPTIONAL] IMAGE SAVE
        if save_flag:
            for i in range(gt_y_data.shape[0]):
                hr_filename = HR_DIR + '/' + str(hr_batch_number).zfill(6) + '.png'
                lr_filename = LRX2_DIR + '/' + str(lr_batch_number).zfill(6) + '.png'
                bi_filename = BICUBIC_DIR + '/' + str(bi_batch_number).zfill(6) + '.png'

                hr_batch_number += 1
                lr_batch_number += 1
                bi_batch_number += 1

                hr = gt_y_data[i]
                lr = lr_y_data[i]
                bi = bi_y_data[i]

                save_image(hr_filename, hr)
                save_image(lr_filename, lr)
                save_image(bi_filename, bi)

    hr_list = torch.FloatTensor(np.concatenate(hr_list))
    lr_list = torch.FloatTensor(np.concatenate(lr_list))
    bi_list = torch.FloatTensor(np.concatenate(bi_list))

    return hr_list, lr_list, bi_list

def load_img(OUTPUT_DIR,expected_totalaug=0,test=False):

    split_list = []
    for i in (sorted(glob(OUTPUT_DIR + '/*'))):
        split_list.append(i)

    if test:
        pass
    else:
        if len(split_list) != expected_totalaug:
            print("Not fittable the number of augmented images !!")
            exit()
        else:
            print("Next Level : Split images Using Window, {} pic".format(len(split_list)))

    return split_list

def resize_image_by_pil(image, scale):
    width, height = image.shape[1], image.shape[0]
    new_width = int(width * scale)
    new_height = int(height * scale)
    method = Image.BICUBIC

    if len(image.shape) == 3 and image.shape[2] == 3:
        image = Image.fromarray(image, "RGB")
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        # the image may has an alpha channel
        image = Image.fromarray(image, "RGB")
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
    else:
        image = Image.fromarray(image.reshape(height, width))
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
        image = image.reshape(new_height, new_width, 1)
    return image


def save_image(filename, image, print_console=True):
    if len(image.shape) >= 3 and image.shape[0] == 1:
        image = image.reshape(image.shape[1], image.shape[2])

    directory = os.path.dirname(filename)
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory)

    image = image.astype(np.uint8)
    if len(image.shape) >= 3 and image.shape[0] == 3:
        image = Image.fromarray(image, mode="RGB")
    else:
        image = Image.fromarray(image)
    imageio.imwrite(filename, image)

    if print_console:
        print("Saved [%s]" % filename)


def set_image_alignment(image, alignment):
    alignment = int(alignment)
    width, height = image.shape[1], image.shape[0]
    width = (width // alignment) * alignment
    height = (height // alignment) * alignment

    if image.shape[1] != width or image.shape[0] != height:
        image = image[:height, :width, :]

    if len(image.shape) >= 3 and image.shape[2] >= 4:
        image = image[:, :, 0:3]

    return image

def convert_rgb_to_y(image, jpeg_mode=False, max_value=255.0):
    if len(image.shape) <= 2 or image.shape[2] == 1:
        return image

    if jpeg_mode:
        xform = np.array([[0.299, 0.587, 0.114]])
        y_image = image.dot(xform.T)
    else:
        xform = np.array([[65.481 / 256.0, 128.553 / 256.0, 24.966 / 256.0]])
        y_image = image.dot(xform.T) + (16.0 * max_value / 256.0)

    return y_image

def convert_rgb_to_ycbcr(image):
    if len(image.shape) < 2 or image.shape[2] == 1:
        return image

    xform = np.array(
        [[65.738 / 256.0, 129.057 / 256.0, 25.064 / 256.0],
         [- 37.945 / 256.0, - 74.494 / 256.0, 112.439 / 256.0],
         [112.439 / 256.0, - 94.154 / 256.0, - 18.285 / 256.0]])

    ycbcr_image = image.dot(xform.T)
    ycbcr_image[:, :, 0] += 16.0
    ycbcr_image[:, :, [1, 2]] += 128.0

    return ycbcr_image


def convert_y_and_cbcr_to_rgb(y_image, cbcr_image):
    print(cbcr_image.shape)
    if len(y_image.shape) <= 2:
        y_image = y_image.reshape[y_image.shape[0], y_image.shape[1], 1]

    if len(y_image.shape) == 3 and y_image.shape[2] == 3:
        y_image = y_image[:, :, 0:1]

    ycbcr_image = np.zeros([y_image.shape[0], y_image.shape[1], 3])
    ycbcr_image[:, :, 0] = y_image[:, :, 0]
    ycbcr_image[:, :, 1:3] = cbcr_image[:, :, 0:2]

    return convert_ycbcr_to_rgb(ycbcr_image)

def convert_ycbcr_to_rgb(ycbcr_image):
    rgb_image = np.zeros([ycbcr_image.shape[0], ycbcr_image.shape[1], 3])  # type: np.ndarray

    rgb_image[:, :, 0] = ycbcr_image[:, :, 0] - 16.0
    rgb_image[:, :, [1, 2]] = ycbcr_image[:, :, [1, 2]] - 128.0
    xform = np.array(
        [[298.082 / 256.0, 0, 408.583 / 256.0],
         [298.082 / 256.0, -100.291 / 256.0, -208.120 / 256.0],
         [298.082 / 256.0, 516.412 / 256.0, 0]])
    rgb_image = rgb_image.dot(xform.T)

    return rgb_image
def get_split_images(image, window_size, stride=None, enable_duplicate=True):
    if len(image.shape) == 3 and image.shape[2] == 1:
        image = image.reshape(image.shape[0], image.shape[1])
    #print(image.shape)
    window_size = int(window_size)
    size = image.itemsize
    height, width = image.shape
    if stride is None:
        stride = window_size
    else:
        stride = int(stride)

    if height < window_size or width < window_size:
        return None

    new_height = 1 + (height - window_size) // stride
    new_width = 1 + (width - window_size) //  stride

    shape = (new_height, new_width, window_size, window_size)
    strides = size * np.array([width * stride, stride, width, 1])
    windows = np.lib.stride_tricks.as_strided(image, shape=shape, strides=strides)
    windows = windows.reshape(windows.shape[0] * windows.shape[1],1, windows.shape[2], windows.shape[3])

    if enable_duplicate:
        extra_windows = []
        if (height - window_size) % stride != 0:
            for x in range(0, width - window_size, stride):
                extra_windows.append(image[height - window_size - 1:height - 1, x:x + window_size:])

        if (width - window_size) % stride != 0:
            for y in range(0, height - window_size, stride):
                extra_windows.append(image[y: y + window_size, width - window_size - 1:width - 1])

        if len(extra_windows) > 0:
            org_size = windows.shape[0]
            windows = np.resize(windows,
                                [org_size + len(extra_windows), windows.shape[1], windows.shape[2], windows.shape[3]])
            for i in range(len(extra_windows)):
                extra_windows[i] = extra_windows[i].reshape([1,extra_windows[i].shape[0], extra_windows[i].shape[1]])
                windows[org_size + i] = extra_windows[i]

    return windows

def GPU_AVAILABLE():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Device:', device)  # 출력결과: cuda
    print('Count of using GPUs:', torch.cuda.device_count())  # 출력결과: 2 (2, 3 두개 사용하므로)
    print('Current cuda device:', torch.cuda.current_device())  # 출력결과: 2 (2, 3 중 앞의 GPU #2 의미)
    return device


