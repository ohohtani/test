import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

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

# Define the DeepLab model
class DeepLab(nn.Module):
    def __init__(self, num_classes):
        super(DeepLab, self).__init__()
        self.num_classes = num_classes

        # Load the pre-trained ResNet-101 model
        resnet = models.resnet101(pretrained=True)

        # Extract the layers from the ResNet model
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Atrous convolutional layer with dilation rate 2
        self.atrous_convolution = nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=2, dilation=2)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.atrous_convolution(x)

        # Upsample the output to match the input size
        x = F.interpolate(x, size=(x.size(2) * 8, x.size(3) * 8), mode='bilinear', align_corners=True)

        return x

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

# Initialize the model and move it to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepLab(num_classes=1).to(device)

# Define the loss function and optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model for 10 epochs
# Train the model for 10 epochs
for epoch in range(10):
    model.train()
    epoch_loss = 0
    for images, masks in dataloader:
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
    for images in test_dataloader:
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
