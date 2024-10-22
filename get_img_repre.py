import argparse


parser = argparse.ArgumentParser(description='Labeling Concepts via CLIP.')
parser.add_argument('--data_name', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
parser.add_argument('--data_path', type=str, default='/data/.../')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--backbone', type=str, default='RN50')

args = parser.parse_args()


import pickle
import torch
import clip
from torchvision.datasets import CIFAR10, CIFAR100

torch.set_num_threads(4)

model, preprocess = clip.load(args.backbone, device='cuda')
if args.data_name == 'cifar10':
    trainset = CIFAR10(root=args.data_path, train=True, download=False, transform=preprocess)
    testset = CIFAR10(root=args.data_path, train=False, download=False, transform=preprocess)
elif args.data_name == 'cifar100':
    trainset = CIFAR100(root=args.data_path, train=True, download=False, transform=preprocess)
    testset = CIFAR100(root=args.data_path, train=False, download=False, transform=preprocess)
model.eval()

if args.mode == 'train':
    dataset = trainset
if args.mode == 'test':
    dataset = testset

Img_Repre_Mat = []
with torch.no_grad():
    for i in range(len(dataset)):
        image = dataset[i][0].unsqueeze(0).cuda()
        image_feature = model.encode_image(image).squeeze()
        Img_Repre_Mat.append(image_feature)

        if i % 10 == 0:
            print(f"Finish Extracting {i} Images!")

with open(f'/data/.../ResCBM/{args.data_name}_image_representation_train.pkl', 'wb') as file:
    pickle.dump(Img_Repre_Mat, file)