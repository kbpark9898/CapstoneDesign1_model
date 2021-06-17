# Chest X-ray classification with pyTorch
흉부 X-ray에 대해 학습시키기 위한 pyTorch resnet, densenet 모델입니다.

본 모델은 2021-1 경희대학교 컴퓨터공학과 캡스톤디자인1 과목에서 수행된 'X-ray 이미지 및 딥러닝을 이용한 병변 인식' 연구의 일환입니다.

(작성자 : 경희대학교 컴퓨터공학과 4학년 2016104122 박기범)

# Requirements
* torch >= 0.4    
* torchvision >= 0.2.2
* opencv-python    
* numpy >= 1.7.3       
* matplotlib       
* tqdm    

# Dataset
[NIH Chest X-ray Dataset](https://www.kaggle.com/nih-chest-xrays/data#Data_Entry_2017.csv) 를 학습시켰습니다. NIH Chest X-ray Dataset은 총 14개의 병증에 대한 라벨을 갖고있으며, 한개의 'No finding' 라벨을 갖고 있습니다. 'No finding' 라벨은 아무런 병증이 없는 정상 흉부 X-ray에 해당하며, 최종적으로 사용한 흉부 X-ray dataset에서 정상 dataset에 해당하는 라벨입니다.

본 dataset이 가지고있는 14개의 병증 라벨은 다음과 같습니다.

* Atelectasis
* Consolidation
* Infiltration
* Pneumothorax
* Edema
* Emphysema
* Fibrosis
* Effusion
* Pneumonia
* Pleural_thickening
* Cardiomegaly
* Nodule Mass
* Hernia

본 dataset은 112,120장의 흉부 X-ray 이미지로 이루어져 있으며 약 86000장을 학습 및 검증에 사용하였고, 약 25000장을 테스트에 사용했습니다.

추가적으로, 캡스톤디자인1의 연구에 사용되는 목적으로 제공받은 심장비대증에 대한 binary classification dataset을 사용했습니다. 이 dataset은 총 2569장의 X-ray로 이루어져 있습니다. 라벨 별 데이터 수는 다음과 같습니다.

* abnormal (심장비대증) : 816
* normal (정상) : 1753

이와 같은 심장 비대증 dataset의 사용 비율은 다음과 같습니다.

* Train : Valid : Test = 7 : 1 : 2





# Sample X-Ray Images

사용된 NIH dataset의 sample은 다음과 같습니다.
<div class="row">
  <div class="column">
    <img src='/sample_xrays/Atelectasis.png' width='250' alt='Atelectasis' hspace='15'>
  </div>
  <div class="column">
    <img src='/sample_xrays/Cardiomegaly_Edema_Effusion.png' width='250' alt='Cardiomegaly | Edema | Effusion' hspace='15'>
  </div>
  <div class="column">
    <img src='/sample_xrays/No Finding.png' width='250' alt='No Finding' hspace='15'>
  </div>
</div>

최종적으로 활용한 심장비대증 dataset은 연구 목적으로 제공받았으므로 sample을 공개하지 않습니다.

# Model 
Master branch에서 다음과 같은 모델을 사용했습니다.

* resnet 18/34/50/101
* densenet 121/169

newData branch에서 다음과 같은 모델을 사용했습니다.

* resnet 50



# Environment

공통적인 학습환경은 다음과 같습니다.

  * CPU  : Intel(R) Core(™) i7-6800K CPU @ 3.40Hhz
  * GPU : NVIDIA TITAN Xp
  * Infra structure
    1. Host OS : Ubuntu 18.04.5 LTS
    2. container
        a. docker  : 20.10.5
        b. pytorch/pytorch image : latest
  *  사용하는 서버에는 상기한 cpu와 gpu가 각각 4개씩 설치되어 있습니다. 기본적으로 하나의 모델을 
      돌리는데 하나의 gpu를 사용했으며, 모든 label에 대한 multi label classification에 대해서는 데이터 
      로드로 인한 병목현상을 줄이기 위해 단일 cpu의 최대 코어수인 6개를 data loader에 부여했습니다.
  * ResNet과 DenseNet의 모델 깊이에 따른 성능 차이를 확인하여 가장 적합한 학습 모델을 찾기 
      위해 ResNet 18/34/50/101, DenseNet 121/169의 6개 모델에 대해 학습시켰으며, NIH dataset에
      대해 multi label classification을 수행했습니다.

Master branch의 모델에서 다음과 같은 학습환경을 사용했습니다.
* claasification : multi-label classification (병증에 대한 모든 라벨)
* Training set : 약 8만장
* Validation set : 약 8천장
* Test set : 약 2만장
* optimizer : adam
* loss function : focal loss
* batch size: 64
* epoch : 50
*  LR : 1e-05
* 특이사항 : NIH dataset에는 특정 병증이 라벨링 된 데이터보다 정상 데이터인 ‘No finding’ 데이터가 압도적으로 많아서 하나의 batch를 생성할 때 실제 질환이 있는 데이터가 선별될 확률이 낮았습니다. 따라서 각 병증별 데이터의 군형을 맞추기 위해 각 라벨별로 최대 10000개의 데이터만 선택될 수 있도록 batch를 구성하여 학습시켰습니다. 또한 no finding label을 포함안 전체의 라벨에 대해 multi label classification이 가능하도록 fully connected layer를 fine tuning 하였습니다.

newData branch의 모델에서 다음과 같은 학습환경을 적용했습니다.
* epoch : 50
* classification : binary classification (normal, abnormal)
* dataset division ratio
    1. Train : Valid : Test = 7 : 1 : 2
* optimizer : adam
* loss function : cross entropy loss
* LR : 1e-05
* LR_scheduler : 7 epoch 마다 현 LR에 0.1을 곱함
* Image size : 224
* augmentation
    1. random horizontal flip
    2. random normalize (mean - 0.5, std = 0.5)


# Training
* ## 개요
  학습은 master branch와 newData branch로 분할하여 진행했습니다. 그 내용은 다음과 같습니다.

* ## Master branch
  master branch는 NIH dataset을 resnet과 densenet 모델들에 학습시키고, 각 모델별로 가장 효율적인 계층을 찾기 위한 브랜치입니다. 이와 동시에 NIH dataset에 대해 pretrain 시킨 feature map을 추출하기 위해 사용되었습니다.


  * ### 모델 구조
    모델의 학습 구조는 기본적으로 다음과 같습니다.
    * layer2
    * layer3
    * layer4
    * fc

    Terminal Code: 
    ```
    python main.py
    ```

  * ### Checkpoint 불러오기
    이 모델에서 생성되는 checkpoint에는 다음과 같은 정보들이 포함됩니다.
    * epochs (number of epochs the model has been trained till that time)
    * model (architecture and the learnt weights of the model)
    * lr_scheduler_state_dict (state_dict of the lr_scheduler)
    * losses_dict (a dictionary containing the following loses)

 
    checkpoint를 불러울때는 다음과 같은 terminal code를 사용합니다. terminal code의 --stage 파라미터는 모델의 학습 계층을 의미합니다. (ex. --stage 4 는 fc를 의미합니다.)
    
    Terminal Code: 
    ```
    python main.py --resume --ckpt checkpoint_file.pth --stage 2
    ```

    Training the model will create a **models** directory and will save the checkpoints in there.


  * ### Testing
    checkpoint가 저장되어있는 dir에서 다음과 같은 terminal code를 통해 test mode를 동작시킵니다.

    Terminal Code: 
    ```
    python main.py --test --ckpt checkpoint_file.pth
    ```
* ## newData branch
  본 branch는 심장 비대증에 대한 binary classification data를 학습시키고 그 결과를 측정하기위한 branch입니다. 

  본 branch 에서 사용된 사전학습 모델은 다음과 같습니다.

  * 최초 학습시 ImageNet에 대해 pretrain되지 않은 상태에서 학습을 진행한 모델 (모델 A)

  * mageNet에 대해 pretrain된 상태에서 학습을 진행한 모델 (모델 B)
  
  본 branch 에서 학습시킨 모델들은 다음과 같습니다.

  * 모델 A에 NIH dataset을 학습시키고, 이에 대해서 새로운 흉부 X-ray dataset을 학습시켜 테스트한 
  경우 (모델 1)

  * 모델 B에 NIH dataset을 학습시키고 이에 대해서 새로운 흉부 X-ray dataset을 학습시켜 테스트한 
  경우 (모델 2)

  * 모델 B에 새로운 흉부 X-ray dataset을 학습시켜 테스트한 경우 (모델 3)

  * 아무것도 학습시키지 않은 모델에 새로운 흉부 X-ray dataset을 학습시켜 테스트한 경우 (모델 4)

  본 branch의 모델을 학습시키기 위해서는 다음과 같은 terminal code를 사용합니다.

  Terminal Code: 
    ```
    python3 new_main.py 
    ```
  모델의 데이터셋 경로변경과 test 실행은 다음의 코드를 수정하여 진행합니다.
  
    ```
      ckpt = torch.load('/root/share/result/resnet50_models/ImageNet_pretrained/stage1_1e-05_50.pth')
      model.load_state_dict(ckpt['model'].state_dict())
      path = '/root/share/origin'
      lr=1e-5


      if parameter=='test':
          model.fc=nn.Linear(num_ftrs, 2)
          ckpt = torch.load('/root/share/result/new_resnet50/ImageNet_based/resnet50_epoch17.pth')
          model.load_state_dict(ckpt['model_state_dict'])
    ```
  모델의 checkpoint 생성은 new_train.py의 다음 코드를 수정하여 진행합니다.
  ```
   if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save({
                    'epoch':epoch,
                    "model_state_dict":best_model_wts,
                    'optimizer_state_dict':optimizer.state_dict(),
                    'loss':loss
                }, '/root/share/result/new_resnet50/ImageNet_based/resnet50_epoch{}'.format(epoch)+'.pth')
                #if you want to use pretrained model, use /root/share/result/new_resnet50/resnet50_epoch*.pth
                print('saved!')
  ```
# Result 
학습 결과는 다음과 같습니다.
  * ### Master branch (ROC_AUC)
    * resnet18 : 0.51
    * resnet34 : 0.52
    * resnet50 : 0.56
    * resnet101 : 0.50
    
    * densenet121 : 0.54
    * densenet169 : 0.53

  * ### newData branch
    앞선 newData branch 설명에서 정의한 모델 1,2,3,4에 대한 학습 결과를 서술합니다.

    * 모델 1 : 69%
    * 모델 2 : 64%
    * 모델 3 : 61%
    * 모델 4 : 44%

