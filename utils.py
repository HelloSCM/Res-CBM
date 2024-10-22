import os
import re
import random
import pickle
import argparse

import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100


def get_args():
    parser = argparse.ArgumentParser(description='CLIP based Concept Bottleneck Model.')
    # file path
    parser.add_argument('--img_path', type=str, default='/data/.../')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--cpt_path', type=str, default='None')
    parser.add_argument('--mdl_path', type=str, default='None')
    parser.add_argument('--candidate_path', type=str, default='None')
    # training info
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--init_lr', type=float, default=0.01)
    parser.add_argument('--decay_step', type=int, default=10)
    parser.add_argument('--decay_rate', type=float, default=0.6)
    parser.add_argument('--init_lr_', type=float, default=0.01)
    parser.add_argument('--decay_step_', type=int, default=10)
    parser.add_argument('--decay_rate_', type=float, default=0.6)
    parser.add_argument('--backbone', type=str, default='RN50')
    # algorithm details
    parser.add_argument('--task', type=str, default='prediction', choices=['prediction', 'discovery'])
    parser.add_argument('--l1_reg', type=float, default=1e-5)
    parser.add_argument('--res_mode', type=str, default='in-process', choices=['in-process', 'post-hoc', 'direct'])
    parser.add_argument('--res_dim', type=int, default=0)
    parser.add_argument('--sim_reg', type=float, default=1.0)
    parser.add_argument('--candidate_num', type=int, default=5)
    
    args = parser.parse_args()
    set_random_seed(args.seed)
    print("The Training Info: ")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    return args


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(args, transform):
    if args.dataset == 'cifar10':
        trainset = CIFAR10(root=args.img_path, train=True, download=False, transform=transform)
        testset = CIFAR10(root=args.img_path, train=False, download=False, transform=transform)

    elif args.dataset == 'cifar100':
        trainset = CIFAR100(root=args.img_path, train=True, download=False, transform=transform)
        testset = CIFAR100(root=args.img_path, train=False, download=False, transform=transform)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    with open(args.cpt_path, 'rb') as cpt_file:
        concept_bank = pickle.load(cpt_file)
        
    print("Finish Loading Dataset...")
    return trainloader, testloader, concept_bank


def prediction_recorder(args, model, metrics):
    result = pd.DataFrame.from_dict(metrics)
    best_epoch = np.argmin(np.array(metrics['Loss']))
    best_result = metrics['Acc'][best_epoch]
    print("========================================")
    print(f"Best Accuracy: {(100.0*best_result):.2f}%")
    
    concept_num = re.findall(r'\d+', args.cpt_path)[-1]
    filename = f"Cpt_{concept_num}_Res_{args.res_dim}_Acc_{best_result:.4f}"
    os.makedirs(f"./results/{args.task}/{args.dataset}/{filename}", exist_ok=True)
    result_path = f"./results/{args.task}/{args.dataset}/{filename}"
    
    args_info = "\n".join([f"{k}: {v}" for k, v in vars(args).items()])
    result_info = (
        f"minimal_loss: {metrics['Loss'][best_epoch]:.4f}\n"
        f"best_accuracy: {(100.0*best_result):.2f}%"
    )
    with open(f"{result_path}/log_info.txt", 'w') as log_file:
        log_file.write(args_info + "\n" + result_info)
            
    result.to_csv(f"{result_path}/metrics.csv", index=False)
    torch.save(model, f"{result_path}/model.pt")
    
    print("Finish Saving Data.")


def discovery_recorder(args, model, metrics, cpt_num, acc_rm):
    result = pd.DataFrame.from_dict(metrics)
    best_epoch = np.argmin(np.array(metrics['Loss']))
    discovered_concept_name = metrics['Concept'][best_epoch]
    best_result = metrics['Acc_org'][best_epoch]
    print("========================================")
    print(f"Discovered Concept: {discovered_concept_name}")
    print(f"Best Accuracy: {(100.0*best_result):.2f}%")

    filename = (
        f"{cpt_num}_Cpt_{discovered_concept_name}_Sim_{metrics['Sim'][best_epoch]:.4f}_"
        f"AccOrg_{metrics['Acc_org'][best_epoch]:.4f}_AccRes_{metrics['Acc_res'][best_epoch]:.4f}_AccRm_{acc_rm:.4f}"
    )
    concept_num = re.findall(r'\d+', args.cpt_path)[-2]
    dir_name = f"Org_{concept_num}_Res_{args.res_dim}_Candidate_{args.candidate_num}_R_{args.sim_reg}"
    os.makedirs(f"./results/{args.task}/{args.dataset}/{dir_name}/{filename}", exist_ok=True)
    result_path = f"./results/{args.task}/{args.dataset}/{dir_name}/{filename}"

    args_info = "\n".join([f"{k}: {v}" for k, v in vars(args).items()])
    result_info = (
        f"after_removing_concept_accuracy: {acc_rm}\n"
        f"discovered_concept: {metrics['Concept'][best_epoch]}\n"
        f"similarity: {metrics['Sim'][best_epoch]:.4f}\n"
        f"minimal_loss: {metrics['Loss'][best_epoch]:.4f}\n"
        f"best_org_accuracy: {metrics['Acc_org'][best_epoch]:.4f}\n"
        f"best_res_accuracy: {metrics['Acc_res'][best_epoch]:.4f}"
    )
    with open(f"{result_path}/log_info.txt", 'w') as log_file:
        log_file.write(args_info + "\n" + result_info)
            
    result.to_csv(f"{result_path}/metrics.csv", index=False)
    torch.save(model, f"{result_path}/model.pt")

    print("Finish Saving Data.")

    return discovered_concept_name