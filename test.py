import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import os

classes = ('COVID', 'Lung_Opacity', 'Normal', 'Viral_Pneumonia')

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5097035, 0.5097035, 0.5097035], std= [0.23124482, 0.23124482, 0.23124482])
])

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model=torch.load('checkpoints/Swin/best.pth')
model.eval()
model.to(DEVICE)

path = 'test/'
testList = os.listdir(path)
for file in testList:
    img = Image.open(path + file).convert('RGB')
    img = transform_test(img)
    img.unsqueeze_(0)
    img = Variable(img).to(DEVICE)
    out = model(img)
    # Predict
    _, pred = torch.max(out.data, 1)
    print('Image Name:{},predict:{}'.format(file, classes[pred.data.item()]))
