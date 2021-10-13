import torch
from func import utils
from func import eval_utils
import numpy as np
import imageio
import os

def test(model_path,lr,bi):

    device = utils.GPU_AVAILABLE()

    if len(lr.shape) == 3 and lr.shape[2] == 1:
        lr_luma = lr.reshape(1,1,lr.shape[0], lr.shape[1])
    if len(bi.shape) == 3 and bi.shape[2] == 1:
        bi_luma = bi.reshape(1,1,bi.shape[0], bi.shape[1])

    lr_luma = torch.FloatTensor(lr_luma).to(device)
    bi_luma = torch.FloatTensor(bi_luma).to(device)

    load_model = torch.load(model_path).to(device)
    load_model.eval()

    with torch.no_grad():
        pred = load_model(lr_luma)
        pred += bi_luma

    pred = pred.detach().cpu().numpy()
    pred = pred.reshape(pred.shape[2],pred.shape[3],1)
    return pred

def main():

    # model load

    load_path = 'save_model/DCSCN_V2_e9_lr0.0001.pt'

    # test image load

    test_dir = 'data/set5'
    test_list = utils.load_img(test_dir,test=True)
    scale_factor = 2
    output_dir = 'test/'

    AVG_PSNR = []
    AVG_SSIM = []
    for i in test_list:
        file_name = os.path.basename(i)
        file_name,ext = os.path.splitext(file_name)

        img = imageio.imread(i)
        lr_img = utils.resize_image_by_pil(img,1/scale_factor)
        bi_img = utils.resize_image_by_pil(lr_img,scale_factor)

        y_img = utils.convert_rgb_to_y(img)
        y_lr_img = utils.resize_image_by_pil(y_img,1/scale_factor)
        y_bi_img = utils.resize_image_by_pil(y_lr_img,scale_factor)
        ycbcr_bi_img = utils.convert_rgb_to_ycbcr(bi_img)

        recon = test(load_path,y_lr_img,y_bi_img)
        recon_rgb = utils.convert_y_and_cbcr_to_rgb(recon,ycbcr_bi_img[:,:,1:3])

        bicubic_rgb = utils.convert_y_and_cbcr_to_rgb(y_bi_img,ycbcr_bi_img[:,:,1:3])

        luma_psnr,luma_ssim = eval_utils.compute_psnr_and_ssim(y_img, recon, 2+scale_factor)
        luma_bipsnr, luma_bissim = eval_utils.compute_psnr_and_ssim(y_img, y_bi_img, 0)

        print("luma_recon : {} / {} luma_bicubic : {} / {}".format(luma_psnr, luma_ssim, luma_bipsnr, luma_bissim))

        org_name = output_dir+file_name + '_org' + ext
        recon_save_name = output_dir+file_name + '_recon' + ext
        bi_save_name = output_dir+file_name + '_bi' + ext

        recon_rgb = np.clip(recon_rgb,0,255)
        bicubic_rgb = np.clip(bicubic_rgb,0,255)

        utils.save_image(org_name,img)
        utils.save_image(recon_save_name,recon_rgb)
        utils.save_image(bi_save_name,bicubic_rgb)

        AVG_PSNR.append(luma_psnr)
        AVG_SSIM.append(luma_ssim)


    avg_psnr = sum(AVG_PSNR)/len(AVG_PSNR)
    avg_ssim = sum(AVG_SSIM)/len(AVG_SSIM)

    print("avg_psnr / avg_ssim : {} / {}".format(avg_psnr,avg_ssim))

if __name__ == "__main__" :
    main()
