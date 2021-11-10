from data_processing import multiprocessing_aug
from func import utils
import os
import torch
import torch.optim as optim
from DCSCN import DCSCN


def main():
    # Checking GPU Available

    # splited 된 그림을 보길 원하시면 batch_picture_save_flag 를 1 로 바꾸시면 됩니다.
    # 경로 : augmented_data/train_sr
    batch_picture_save_flag = 0
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Device:', device)  # 출력결과: cuda
    print('Count of using GPUs:', torch.cuda.device_count())  # 출력결과: 2 (2, 3 두개 사용하므로)
    print('Current cuda device:', torch.cuda.current_device())  # 출력결과: 2 (2, 3 중 앞의 GPU #2 의미)

    # Configure Data Augmentation

    DATA_DIR = ['data/bsd200', 'data/yang91']
    OUTPUT_DIR = 'augmented_data/train_org/'

    utils.SR_FOLDER_GENERATE()
    expected_totalaug = multiprocessing_aug(DATA_DIR,OUTPUT_DIR)

    # Split Parameters

    BICUBIC_DIR = 'augmented_data/train_sr/LRBICUBIC'
    LRX2_DIR = 'augmented_data/train_sr/LRX2'
    HR_DIR = 'augmented_data/train_sr/HR'

    lr_batch_size = 32
    scale = 2
    train_list = utils.load_img(OUTPUT_DIR,expected_totalaug)
    HR_LIST, LR_LIST, BI_LIST = utils.build_data(train_list, lr_batch_size, scale, BICUBIC_DIR, LRX2_DIR, HR_DIR, batch_picture_save_flag)

    # TORCH BATCH DATASET
    batch_size = 20
    train_dataset = torch.utils.data.TensorDataset(LR_LIST, HR_LIST, BI_LIST)
    data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )

    if len(HR_LIST) == len(LR_LIST) == len(BI_LIST):
        total_batch_num = len(data_loader)
        pass
    else:
        print("ERROR : NOT MATCH NUMBER OF PAIR BATCH DATA")
        exit()

    MODEL = DCSCN().to(device)
    print(MODEL)


    lr = 1e-4
    total_epochs = 1000
    model_path = 'save_model'
    optimizer = optim.Adam(MODEL.parameters(),lr = lr)
    loss_func = torch.nn.MSELoss().to(device)

    for epochs in range(total_epochs):
        avg_loss = 0.0
        batch_num = 0
        for LR,HR,BI in data_loader:

            recon = MODEL(LR.to(device))
            recon += BI.to(device)
            loss = loss_func(recon, HR.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss / batch_size

            if batch_num % 1000 == 0 :
                print("{}/{} Training....batch_loss {}!".format(batch_num,total_batch_num, avg_loss/batch_num))

            batch_num += 1


        avg_loss = avg_loss / total_batch_num
        print("Epoch {}/{} loss {}".format(epochs+1, total_epochs, avg_loss))
        if epochs % 10 == 0:
            save_model_path = model_path + "/DCSCN_V2_e{}_lr{}_loss{:4}.pt".format(epochs+1,lr,loss)
            torch.save(MODEL, save_model_path)

            print("SAVE MODEL EPOCH {}".format(epochs))


    # Last save

    save_model_path = model_path + "/DCSCN_V2_e{}_lr{}.pt".format(epochs+1, lr)
    torch.save(MODEL, save_model_path)

    print("LAST SAVE MODEL EPOCH {}".format(epochs+1))


if __name__ == "__main__":
    main()
