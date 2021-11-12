# 2021-2.ComputerVision_DCSCN_pytorch
#### [[Youtube](https://youtu.be/OpgsHyngR_A)] [[EvalAI](http://203.250.148.129:3088/web/challenges/challenge-page/81/overview)]

## Introduction
![image](https://user-images.githubusercontent.com/11037567/137428754-da0497d6-4d09-4ef4-9ba3-0c753b4e9a2e.png)

---

## Set Up Environments
- Windows / pytorch-nightly
- RTX 3090 
- anaconda3(python3.8)

---

## Submission Making
빠른 제출을 위해 제출용코드 제공하고 있습니다.

- SET5는 이미지 화질 개선 테스크에서 5개의 테스트 데이터셋입니다!
- SET5의 순서와 동일하도록 submission_image 풀더에 넣어주세요!
<p align="center">
  <img width="417" alt="3634646346" src="https://user-images.githubusercontent.com/11037567/141294974-247af649-e3e8-4085-8cb4-a8efd2b459ef.png">
</p>

1. sumbission_making/submission_image 풀더에 있는 원본 이미지를 삭제하고, SET5의 복원된 이미지를 넣는다. 
2. $python make_submission.py 
3. "submit.json" 이름으로 파일이 생성된다 (오류가 나면 해당 파일이 삭제됩니다. 오류가 나면 복원된 영상 5개가 풀더 안에 들어가있는지, SET5와 순서가 동일한지 확인해보세요.)

- 정상적으로 Submit.json 이 생성된 모습
  
<img width="329" alt="235235253" src="https://user-images.githubusercontent.com/11037567/141408883-78bc2324-aced-4ee5-bddc-57865e3117f6.PNG">

--- 

## Train
- Train_Data: ( bsd200 + yang91 ) * 8 (augmentation)
- Image_batch_size(Low Resolution size) = 32
- Batch_size = 20
- Scale_factor = 2
- Learning rate = 1e-4, epochs = 1000
- Loss : MSEloss, Optimizer : Adam
```shell
# 생성되는 splited 된 luma 영상을 보길 원하시면 train.py의 batch_picture_save_flag를 1 로 바꾸시면 됩니다.
# 저장 경로 : augmented_data/train_sr
python train.py
```
![image](https://user-images.githubusercontent.com/11037567/137428803-233efc3c-d790-4511-bb5e-47805b6fa31b.png)


## Test
- Test_data : SET5 
- Model : /save_model/DCSCN_V2_e100_lr0.0001.pt
```shell
python test.py
```
![image](https://user-images.githubusercontent.com/11037567/137428827-80f0f665-617d-48cf-b140-c6952f9cd1ed.png)
 
---

### Reference
[1] J.Yamanaka et al, "Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network"  
[2] https://github.com/sejong-rcv/EvalAI-Starters  
