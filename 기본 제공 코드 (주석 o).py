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



class SatelliteDataset(Dataset):
    def __init__(self, csv_file, transform=None, infer=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            return image

        mask_rle = self.data.iloc[idx, 2]
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask


transform = A.Compose(
    [   
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2()
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
