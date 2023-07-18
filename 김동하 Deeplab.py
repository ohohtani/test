import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import albumentations as A

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def rle_decode(mask_rle, shape):    # rle_decode라는 함수를 정의한다. 인자로는 rle 방식으로 압축된 mask와 원래 형태의 이미지(shape)를 받는다.
    s = mask_rle.split()            # maks_rle(rle 방식으로 압축된 마스크)를 공백을 기준으로 분할하여 s 리스트에 저장한다. (split이 그런 역할임 ㅇㅇ)
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]   # s 리스트에서 짝수 인덱스와 홀수 인덱스를 분리하여 각각 starts에 위치 lengths에 길이로 저장한다.
                                                                                     # np.asarray 는 시작 위치와 길이를 정수 배열로 변환하는 역할을 해준다.
    starts -= 1  # RLE 에서는 인덱스가 1부터 시작한다고 한다. 배열의 인덱스는 0부터 시작하기 때문에 배열의 모든 요소에 1을 빼준다.
    ends = starts + lengths  # 끝 위치 = 시작 위치 + 길이
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)  # 이미지 1차원 배열 생성
    for lo, hi in zip(starts, ends):     # starts 와 ends 배열을 순회하면서 img 배열에서 해당하는 구간을 1로 설정(해당 run의 위치를 표시하는 것)한다.  ? 무슨 말일까 : lo, hi 는 각 run의 시작위치와 끝 위치를 나타냄
        img[lo:hi] = 1                   
    return img.reshape(shape)            # 아무튼 img 배열을 주어진 형태(reshape)로 다시 변형하여 복원된 마스크를 반환한다.

# RLE 인코딩 함수 , 디코딩을 했으니 인코딩을 해줘야한다. 참고로    [디코딩 = 압축된 데이터를 원래 형태로 해독, 인코딩 = 원래의 데이터를 다른 형식으로 변환]
def rle_encode(mask):  # 그니까 인코딩 정의에 의해서 인자는 그냥 원래 데이터를 받는 거임 (아마?)
    pixels = mask.flatten()                              #  mask 배열을 1차원으로 펼쳐 pixels에 저장한다. -> 마스크의 모든 값을 하나의 1차원 배열로 변환한다. (1차원이 돼야 RLE 압축이 된다)
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# Define the SatelliteDataset class
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
        
# Define the data transformations
transform = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2()
    ]
)

# Load the training data
dataset = SatelliteDataset(csv_file='./train.csv', transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

# DeepLab의 ASPP 블록 정의
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        # ASPP에 사용할 다양한 dilated convolution 크기들을 정의
        dilations = [1, 6, 12, 18]
        # dilated convolution을 4개 적용 (합산)하여 채널 수를 줄여줌
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=dilations[0], dilation=dilations[0])
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=dilations[1], dilation=dilations[1])
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=dilations[2], dilation=dilations[2])
        self.conv5 = nn.Conv2d(in_channels, out_channels, 3, padding=dilations[3], dilation=dilations[3])
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1_output = nn.Conv2d(out_channels * 5, out_channels, 1)

    def forward(self, x):
        # 각각의 dilated convolution을 적용하고 합침
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = self.conv5(x)
        global_avg_pool = self.global_avg_pool(x)
        global_avg_pool = self.conv1(global_avg_pool)
        # 모든 결과를 합침
        output = torch.cat([conv1, conv2, conv3, conv4, conv5, global_avg_pool], dim=1)
        # 1x1 convolution을 통해 최종 결과를 반환
        output = self.conv1x1_output(output)
        return output

# DeepLab 모델 정의
class DeepLab(nn.Module):
    def __init__(self, num_classes):
        super(DeepLab, self).__init__()
        self.num_classes = num_classes
        # 인코더 부분과 디코더 부분에 사용할 ASPP 블록 정의
        self.aspp = ASPP(in_channels=512, out_channels=256)
        # 인코더 부분의 층을 정의 (먼저 기반 코드에서 이미 정의되어 있음)
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
        # 디코더 부분의 층을 정의 (먼저 기반 코드에서 이미 정의되어 있음)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        # 최종 출력에 사용할 convolution 층 정의
        self.conv_last = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # 인코더 부분의 순전파 수행 (기반 코드와 동일)
        conv1 = self.dconv_down1(x)
        x = nn.MaxPool2d(2)(conv1)
        conv2 = self.dconv_down2(x)
        x = nn.MaxPool2d(2)(conv2)
        conv3 = self.dconv_down3(x)
        x = nn.MaxPool2d(2)(conv3)
        x = self.dconv_down4(x)
        # ASPP 블록을 사용하여 다양한 컨텍스트 정보 캡처
        x = self.aspp(x)
        # 디코더 부분의 순전파 수행 (기반 코드와 동일)
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        # 최종 출력 층 적용
        out = self.conv_last(x)
        return out

# Initialize the model and move it to the GPU if available
model = DeepLab(num_classes=1).to(device)

# Define the loss function and optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model for 10 epochs

for epoch in range(10):
    model.train()
    epoch_loss = 0
    for images, masks in tqdm(dataloader):
        images = images.float().to(device)
        masks = masks.float().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(1), masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader)}')

# Load the test data
test_dataset = SatelliteDataset(csv_file='./test.csv', transform=transform, infer=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

# Evaluate the model on the test data
with torch.no_grad():
    model.eval()
    result = []
    for images in tqdm(test_dataloader):
        images = images.float().to(device)

        outputs = model(images)
        masks = torch.sigmoid(outputs).cpu().numpy()
        masks = np.squeeze(masks, axis=1)
        masks = (masks > 0.35).astype(np.uint8)

        for i in range(len(images)):
            mask_rle = rle_encode(masks[i])
            if mask_rle == '':
                result.append(-1)
            else:
                result.append(mask_rle)

# Save the predictions to a CSV file
submit = pd.read_csv('./sample_submission.csv')
submit['mask_rle'] = result
submit.to_csv('./submit.csv', index=False)
