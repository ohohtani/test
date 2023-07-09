# 헷갈릴까봐 주석을 달아보았어요. 참고하세요

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

dataset = SatelliteDataset(csv_file='./train.csv', transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)


# U-Net의 기본 구성 요소인 Double Convolution Block을 정의합니다.
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

# 간단한 U-Net 모델 정의
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   

        x = self.dconv_down4(x)

        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out


# model 초기화
model = UNet().to(device)

# loss function과 optimizer 정의
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# training loop
for epoch in range(10):  # 10 에폭 동안 학습합니다.
    model.train()
    epoch_loss = 0
    for images, masks in tqdm(dataloader):
        images = images.float().to(device)
        masks = masks.float().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks.unsqueeze(1))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader)}')



test_dataset = SatelliteDataset(csv_file='./test.csv', transform=transform, infer=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)



with torch.no_grad():
    model.eval()
    result = []
    for images in tqdm(test_dataloader):
        images = images.float().to(device)
        
        outputs = model(images)
        masks = torch.sigmoid(outputs).cpu().numpy()
        masks = np.squeeze(masks, axis=1)
        masks = (masks > 0.35).astype(np.uint8) # Threshold = 0.35
        
        for i in range(len(images)):
            mask_rle = rle_encode(masks[i])
            if mask_rle == '': # 예측된 건물 픽셀이 아예 없는 경우 -1
                result.append(-1)
            else:
                result.append(mask_rle)


submit = pd.read_csv('./sample_submission.csv')
submit['mask_rle'] = result

submit.to_csv('./submit.csv', index=False)
