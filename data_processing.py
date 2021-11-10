from glob import glob
from func.data_augment import augment_data
import multiprocessing
from multiprocessing import Pool

def multiprocessing_aug(DATA_DIR,OUTPUT_DIR):

    NUM_CPU = multiprocessing.cpu_count()
    print("NUM_CPU : ", NUM_CPU)

    file_list = []
    file_num = 0
    for i in DATA_DIR:
        print("Augmentation Target : ", i)
        for j, file in enumerate(sorted(glob(i+'/*'))):
            file_list.append([file,file_num,OUTPUT_DIR])
            file_num += 1

    total_auglen = len(file_list)*8
    print(len(file_list)," LIST UP COMPLETED!!")

    pool = Pool(NUM_CPU)
    pool.map(augment_data,file_list)
    pool.close()
    pool.join()

    print("augmentation done..!!")
    return total_auglen
