# 헷갈릴까봐 주석을 달아보았어요. 참고하세요
# 오류 패치노트 
  # 일단 컴파일 오류 첫번째는 train.csv 로드 과정이었다. 동하 말대로 드라이브에 파일 올리고 실행했더니 컴파일은 되나 workers의 개수가 너무 많다는 오류가 뜬다 4를 2로 줄이니 어찌저찌 컴파일은 됐다.
  # workers가 cpu 코어 개수에 따라 상하향 가능하며 다다익선이다. 내 cpu의 코어는 8인데 4로만 설정해도 계속 오류가 난다. 2로 줄이면 성능 저하에 문제가 있을 듯한데 흠.. 답답하다
  # 아무튼 workers 를 2로 한다 치고 통과하고 나면 for images, masks in tqdm(dataloader): 부분에 빨간 줄이 뜨는데 이건 정확한 이유 파악 못함 tlqkf
  # 넘어가서 그 다음 오류는 test.csv 를 불러오는 과정이다. 이 놈은 왜 업로드를 해도 안 불러와지는지 모르겠다. sample 머시기 파일도 마찬가지. 너무 졸려서 내일 재시도 예정
  # 아무튼 여기까지의 오류를 해결하면 성공적인 컴파일이 될 것이다. 참고로 아시겠지만 컴파일 성공된다고 사진이 뽝 출력되는게 아니라 
  # 컴파일 완료 후에 제공된 폴더에 있는 사진을 한 두장씩 불러와 출력해보면서 이미지 세분화가 성공적으로 됐는지 확인해보는 것. 따라서 테스트로 사진 불러오는 건 따로 작성해야 한다. 근데 아마 이건 걍 별거없음

import os    # 파일 디렉토리 관련
import cv2  
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader    # 사용자 정의 데이터셋 클래스를 생성할 때 사용 -> 이미지, 텍스트 등 다양한 유형의 데이터셋 처리 가능
from torchvision import transforms   # 이미지 전처리를 편리하게 수행 가능

from tqdm import tqdm    # 진행상황을 시각화 해준다
import albumentations as A   # 이미지 처리 작업을 위한 다양한 도메인 지원
from albumentations.pytorch import ToTensorV2   # albumentations 를 통해 수행된 이미지 변환을 텐서 형식으로 바꿀 수 있다

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU를 사용할 수 있는 경우에는 사용하고, 그렇지 않으면 CPU를 사용한다 (GPU 사용시 작업수행이 굉장히 빠른데 이거하려면 엔비디아 드라이브 다운받아야 돼서
                                                                       # 꽤나 귀찮지만 여유가 있다면 하면 더 좋긴하다. 본인스펙 RTX 3060에  cpu 라이젠 7 5800X인데 드라이브 다운로드 한시간 넘게 걸렸음)


# RLE 디코딩 함수 , RLE는 그냥 데이터 압축 방식 중 하나라고 생각하면 된다
def rle_decode(mask_rle, shape):    # rle_decode라는 함수를 정의한다. 인자로는 rle 방식으로 압축된 mask와 원래 형태의 이미지(shape)를 받는다.
    s = mask_rle.split()            # maks_rle(rle 방식으로 압축된 마스크)를 공백을 기준으로 분할하여 s 리스트에 저장한다. (split이 그런 역할임 ㅇㅇ)
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]   # s 리스트에서 짝수 인덱스와 홀수 인덱스를 분리하여 각각 starts에 위치 lengths에 길이로 저장한다.
                                                                                     # np.asarray 는 시작 위치와 길이를 정수 배열로 변환하는 역할을 해준다.
    starts -= 1  # RLE 에서는 인덱스가 1부터 시작한다고 한다. 배열의 인덱스는 0부터 시작하기 때문에 배열의 모든 요소에 1을 빼준다.
    ends = starts + lengths  # 끝 위치 = 시작 위치 + 길이
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)  # 이미지 1차원 배열 생성
    for lo, hi in zip(starts, ends):     # starts 와 ends 배열을 순회하면서 img 배열에서 해당하는 구간을 1로 설정한다.  ? 무슨 말일까
        img[lo:hi] = 1                   
    return img.reshape(shape)            # 아무튼 img 재열을 주어진 형태(reshape)로 다시 변형하여 복원된 마스크를 반환한다.

# RLE 인코딩 함수 , 디코딩을 했으니 인코딩을 해줘야한다. 참고로    [디코딩 = 압축된 데이터를 원래 형태로 해독, 인코딩 = 원래의 데이터를 다른 형식으로 변환]
def rle_encode(mask):  # 그니까 인코딩 정의에 의해서 인자는 그냥 원래 데이터를 받는 거임 (아마?)
    pixels = mask.flatten()                              #  mask 배열을 1차원으로 펼쳐 pixels에 저장한다. -> 마스크의 모든 값을 하나의 1차원 배열로 변환한다. (1차원이 돼야 RLE 압축이 된다)
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)





# 열심히 읽어주셔서 감사하나 사실 여기까지는 수정할 필요가 없다. 우리는 성능 향상이 목적이기 때문에 이정도만 이해하고 밑을 수정해야 한다.





class SatelliteDataset(Dataset):     # PyTorch에서 제공하는 Dataset 클래스를 상속받는 클래스 SatelliteDataset 이다. Dataset이란 놈은 데이터셋을 표현하고 로드하는 기능을 제공한다.
    def __init__(self, csv_file, transform=None, infer=False):    # init - 알다시피 초기화. transform은 전처리, infer는 추론(학습)이다. 예측 등에 사용 된다.
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer

    def __len__(self):   # 데이터셋의 샘플 개수를 반환한다. so, 데이터셋의 크기를 알려주는 역할을 할 수 있는데
        return len(self.data)  # 이를 통해 학습 과정에서 샘플 개수를 반환하여 반복 횟수를 결정하거나 배치(batch)처리 등에 활용 가능하다

    def __getitem__(self, idx):  # 특정 인덱스에서 샘플을 가저오는 역할
        img_path = self.data.iloc[idx, 1]   
        image = cv2.imread(img_path)    # 이미지 파일을 읽고
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 색상체계를 BGR에서 RGB로 변환시킴 (희한하게 기본값이 BGR임)
        
        if self.infer:    # infer 가 True 라면 (추론 모드 On 이라면)
            if self.transform:   #  + transform 이 존재한다면 (전처리 모드 On 이라면)
                image = self.transform(image=image)['image']  # image 에 전처리를 적용한다.  (따라서 추론모드(infer)와 transform이 모두 On 상태여야 전처리를 진행 + 반환함을 알 수 있다.)
            return image  

        mask_rle = self.data.iloc[idx, 2]  # 그럼 여기는 추론모드 off 상태이다, 데이터셋에서 idx번째 행의 2번째 열에 해당하는 마스크 정보를 가져온다. 
                                           # 액셀 파일을 확인해보니 mask_rle가 2번 인덱스 열에 있다. 그래서 그런가 보다
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))    # 마스크를 디코딩한다. 그리고 이미지의 크기와 일치하도록 맞춰준다. (shape[0] = 높이, shape[1] = 너비)

        if self.transform:  # 전처리 모드 On 일 시 (추론 모드와 별개)
            augmented = self.transform(image=image, mask=mask)   # 전처리의 결과를 augmented에 저장한다. 
            image = augmented['image'] # 전처리가 적용된 image 업데이트
            mask = augmented['mask']   # 전처리가 적용된 mask 업데이트

        return image, mask   


# 간단 요약 : SateliteDataset이라는 클래스를 정의했으며, 데이터셋을 초기화, 크기 반환, 특정 인덱스의 샘플 가져오기 등을 수행했다.
#            따라서 100%는 아니지만 높은확률로 아직까지는 건드릴 코드가 없다. 


transform = A.Compose(                #  전처리 과정 , 아무래도 여기부터 건드려야 할 것
    [   
        A.Resize(224, 224),           #  이미지 크기를 224 x 224 로 조정
        A.Normalize(),                #  이미지를 정규화 (그냥 학습 및 추론을 돕는 일반적인 단계)
        ToTensorV2()                  #  이미지를 텐서 형태로 변환 (파이토치는 이미지를 텐서로 처리하기에 이 과정은 필수)
    ]
)

dataset = SatelliteDataset(csv_file='./train.csv', transform=transform)   # 위에 적용한 전처리로 SatelliteDataset 클래스를 초기화
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)  # shuffle은 데이터 무작위로 섞기, num_workers는 데이터 처리 작업자인데 다다익선임 cpu코어 개수에 따라 상하향 가능
                                                                              # 컴파일에서 오류 뜨는 부분임 개빡침 

# U-Net 틀 부분임. 건드릴 필요 없음
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),   # Padding = 1의 의미는 그냥 입출력의 크기를 같게 하겠다 라는 뜻
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

# 간단한 U-Net 모델 정의
class UNet(nn.Module):                                
    def __init__(self):                               # 초기화 부분입니다
        super(UNet, self).__init__()
        self.dconv_down1 = double_conv(3, 64)         # 채널 수 3, 출력 수 64
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)                                                          # 한 번쯤 들어보셨을 맥스 풀링 연산(다운 샘플링) (입력의 크기를 절반으로 줄인다 = 해상도를 절반으로 줄인다)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        # 양방향 업샘플링 (입력의 크기를 두배로 늘림 = 해상도를 2배로 늘린다)

        self.dconv_up3 = double_conv(256 + 512, 256)          # 업샘플링 경로에서 사용되는 블록
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, 1, 1)                  # 최종적으로 채널의 수가 1이되며 입력이미지와 같은 해상도를 지닌다. (이해가 안될시 U-Net 구성 이미지 참고)
                                                              # https://www.researchgate.net/publication/332102788/figure/fig2/AS:753454205116418@1556648889776/Modified-U-net-network-structure.jpg

    def forward(self, x):                                     # forward(순전파) 클래스 = 입력 데이터를 모델에 통과시켜 예측값을 계산하는 과정이다
        conv1 = self.dconv_down1(x)                           # 입력 데이터 x를 self.dconv_down1로 전달하여 첫번째 double convolution 블록을 적용(위에 있는거), 이 값을 conv1에 저장
        x = self.maxpool(conv1)                               # conv1에 맥스 풀링(해상도 타노스)을 적용하여 x에 다시 저장

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)          

        x = self.dconv_down4(x)                               # 까지 반복 후

        x = self.upsample(x)                                  # 다시 해상도를 늘리면서 올라간다 그니까 그냥 U-Net 모양 생각하면 된다 내려갔다 올라오는 그 과정을 코드로 나타낸 것.
        x = torch.cat([x, conv3], dim=1)                      # 위에 써있는 '간단한 U-Net 모델 정의'가 힌트 아닐까?(내가쓴거 아님) U-Net을 더 깊고 복잡하게 만들면 성능 향상이 가능한데,
                                                              # 일부러 간단한 구조를 줌으로써 우리보고 수정하라는 힌트를 준거 아닌가? 라고 혼자 생각해보았다.
        x = self.dconv_up3(x)                                 # 어차피 U-Net 제외하면 다 외부 데이터셋이다.
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   

        x = self.dconv_up1(x)

        out = self.conv_last(x)                               # 최종 예측값 계산, 최종 출력 수는 64->1이 된다. 위쪽 self.conv_last 참고

        return out


# model 초기화, 분명히 이 부분이 하이라이트다. 모델의 학습에 관여하기 때문에 가장 중요한 부분이라 생각한다.
model = UNet().to(device)                                                 # 위쪽에서 정의한 U-Net 인스턴스 생성, 모델을 GPU로 이동시킴.  (device) <- 이게 GPU로 이동시키는 명령이라 생각하면 됨. 이유는 성능 향상? 인듯

# loss function과 optimizer 정의
criterion = torch.nn.BCEWithLogitsLoss()                                 # 손실함수를 BCEWith머시기로 설정하였는데, 이진 분류에 적합하긴 하나 다른 손실 함수를 고려해보는 것도 방법이 될 수 있다. (애매)                        
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)               # lr = 학습률. 학습률이 0.001이라는 것은 가중치 업데이트마다 현재 가중치 x 0.001을 하는 것인데 그럼 엄청 미세하게 조정 되는거니까
                                                                         # 학습이 더 많은 반복을 통해 수렴할 수 있는 것이다. 학습률 조정 또한 성능 향상의 키가 될 수 있다. 이것도 소소익선은 아니다

# training loop
for epoch in range(10):                                                  # 학습 10회 진행, 학습 횟수는 다다익선인줄 알았으나 이런저런 이유로 최적의 학습횟수를 직접 찾아나가는 것이 좋다. 보통 10, 100, 1000회로 하더라
    model.train()                                                        # 모델을 학습 모드로 설정
    epoch_loss = 0                                                       # 손실 값 저장하기 위한 변수, 초기화
    for images, masks in tqdm(dataloader):                               # dataloader로부터 이미지, 마스크 배치를 가져온다
        images = images.float().to(device)                               # 이미지 배치를 float형태로 변환하고 GPU로 이동한다. 
        masks = masks.float().to(device)                                 # 마스크 배치를 float형태로 변환하고 GPU로 이동한다. 

        optimizer.zero_grad()                                            # 옵티마이저 기울기 0으로 초기화
        outputs = model(images)                                          # 모델에 이미지 배치를 전달하여 outputs(예측값)을 얻는다.
        loss = criterion(outputs, masks.unsqueeze(1))                    # 예측값과 실제 마스크를 비교하여 손실(loss) 계산.
        loss.backward()                                                  # 역전파를 통해 기울기 계산
        optimizer.step()                                                 # 계산된 기울기를 사용해 모델의 파라미터(학습 과정에서 조정되는 가중치와 편향) 업데이트

        epoch_loss += loss.item()                                        # 현재 배치의 손실값을 epoch_loss에 더한다.

    print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader)}')        # 현재 에폭(학습)의 평균 손실값 출력.  (len(dataloader)는 배치의 개수)

# 변경 고려해 볼 사항 : U-Net 구조, 손실 함수의 종류, 학습률, 학습 횟수. + 그 외, 옵티마이저는 잘 모르겠는데 가만히 놔두는 건 아닐듯

test_dataset = SatelliteDataset(csv_file='./test.csv', transform=transform, infer=True)   # 위에 쓴거랑 구조 같음
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)   # shuffle=False 섞 x -> 순서대로 가져옴


# 추론 & 예측 결과 저장 단계
with torch.no_grad():                                                    # no_grad = 기울기 계산 비활성화 -> 추론 단계에서는 모델 가중치 업데이트 필요없기 때문 -> 계산 자원 절약
    model.eval()                                                         # 모델을 추론 모드로 전환.
    result = []                                                          # 예측 결과 저장할 리스트임
    for images in tqdm(test_dataloader):                                 # 테스트 데이터로더에서 이미지 가져옴
        images = images.float().to(device)                               # 가져온 이미지 float형으로 변환 & GPU 이동
        
        outputs = model(images)                                          # 모델에 이미지를 입력하여 예측 결과를 얻음
        masks = torch.sigmoid(outputs).cpu().numpy()                     # 모델의 출력을 sigmoid를 통해 확률 값으로 변환 시키고 CPU로 데이터를 이동시켜서 NumPy 배열 형태로 변환한다
        masks = np.squeeze(masks, axis=1)                                # 출력된 마스크에서 차원을 한단계 줄인다. (채널 차원을 제거하는 것인데, 최종 예측 마스크는 단일 채널의 이진 이미지로 표현되어야 하기 때문)
        masks = (masks > 0.35).astype(np.uint8) # Threshold = 0.35       # 마스크에 대해 임계값을 적용하여 이진화한다. 기준은 0.35
                                                                         # 임계값 = 어떤 값을 기준으로 이진화를 수행하는데 사용되는 값이다 기본 설정 기준이 0.35니까 이것보다 작으면 0, 크면 1로 분류된다
                                                                         # 이 값 또한 당연히 우리가 최적의 값을 찾아내야 한다. 
        
        for i in range(len(images)):                                     # 이미지 개수만큼 반복한다
            mask_rle = rle_encode(masks[i])                              # 예측된 마스크를 rle 방식으로 인코딩
            if mask_rle == '':                                           # 예측된 건물 픽셀이 아예 없는 경우 -1      
                result.append(-1)
            else:                                                        # 예측된 건물이 픽셀이 있다면?
                result.append(mask_rle)                                  # 그대로 인코딩 결과를 결과 리스트에 추가


submit = pd.read_csv('./sample_submission.csv')                          # 그냥 제출용
submit['mask_rle'] = result

submit.to_csv('./submit.csv', index=False)
