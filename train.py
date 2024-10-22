import torch
import torch.nn as nn
import torch.optim as optim

from model import CLIPFeaturizer
from algorithm.res_cbm import ResCBM
from algorithm.res_discovery_cbm import ResDiscoveryCBM
from utils import *
from discovery import *

import warnings
warnings.filterwarnings('ignore')


def train(args, train_loader, test_loader, feature_extractor, model, criterion, optimizer, scheduler):
    metrics = {'Loss':[], 'Acc': []}
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        pred_type = 'org'

        if args.res_mode == 'in-process' or (args.res_mode == 'post-hoc' and epoch >= round(0.6 * args.epochs)):
            pred_type = 'res'
        if args.res_mode == 'direct':
            pred_type = 'all'
            
        for batch_id, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.cuda(), labels.cuda()
            feas = feature_extractor(imgs)
            batch_result = model.update(feas, labels, criterion, optimizer, args.res_mode, pred_type)
            running_loss += batch_result
        loss = running_loss / batch_id
        scheduler['org'].step()
        scheduler['res'].step()
            
        print(f"==================== Epoch [{epoch+1}/{args.epochs}] ====================")
        print(f"Train - Loss: {loss:.4f}")
        
        if epoch == args.epochs - 1:
            acc = eval(args, model, feature_extractor, test_loader, pred_type)
        else:
            acc = 0.0
        
        metrics['Loss'].append(loss)
        metrics['Acc'].append(acc)

    print("Finish Training!")
        
    return model, metrics


def eval(args, model, feature_extractor, test_loader, pred_type='res'):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.cuda(), labels.cuda()
            feas = feature_extractor(imgs)
            outputs = model.predict(feas, pred_type)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
    acc = correct / total
    print(f"Test - Accuracy: {(100.0*acc):.2f}%")
    
    return acc


def main():
    args = get_args()
    feature_extractor = CLIPFeaturizer(args.backbone).cuda()
    train_loader, test_loader, concept_bank = load_data(args, feature_extractor.preprocess)

    feature_dim = feature_extractor.clip_featurizer.visual.output_dim
    class_num = len(torch.unique(torch.tensor(train_loader.dataset.targets)))
    
    if args.task == 'prediction':
        model = ResCBM(class_num=class_num, res_dim=args.res_dim, concept_bank=concept_bank, reg=args.l1_reg)
        model = model.cuda()
        
        criterion_ce = nn.CrossEntropyLoss()
        optimizer_org = optim.Adam(model.parameters(), lr=args.init_lr)
        optimizer_res = optim.Adam(model.parameters(), lr=args.init_lr_)
        scheduler_org = optim.lr_scheduler.StepLR(optimizer_org, step_size=args.decay_step, gamma=args.decay_rate)
        scheduler_res = optim.lr_scheduler.StepLR(optimizer_res, step_size=args.decay_step_, gamma=args.decay_rate_)
        
        criterion = criterion_ce
        optimizer = {'org': optimizer_org, 'res': optimizer_res}
        scheduler = {'org': scheduler_org, 'res': scheduler_res}

        model, metrics = train(args, train_loader, test_loader, feature_extractor, model, criterion, optimizer, scheduler)
        prediction_recorder(args, model, metrics)


    elif args.task == 'discovery':
        with open(args.candidate_path, 'rb') as cpt_file:
            candidate_concept_bank = pickle.load(cpt_file)
        for cpt in concept_bank.keys():
            if cpt in candidate_concept_bank:
                del candidate_concept_bank[cpt]
                
        pt_mdl = torch.load(args.mdl_path)
        unknown_cpts = pt_mdl.unknown_concepts

        for cpt_num in range(unknown_cpts.shape[0]):
            model = ResDiscoveryCBM(class_num=class_num, concept_bank=concept_bank, unknown_concepts=unknown_cpts, reg_l1=args.l1_reg, reg_sim=args.sim_reg)      
            model = update_init_weights(model, pt_mdl)
            model = model.cuda()
        
            criterion = nn.CrossEntropyLoss()
            optimizer_org = optim.Adam(model.parameters(), lr=args.init_lr)
            optimizer_res = optim.Adam(model.parameters(), lr=args.init_lr_)
            scheduler_org = optim.lr_scheduler.StepLR(optimizer_org, step_size=args.decay_step, gamma=args.decay_rate)
            scheduler_res = optim.lr_scheduler.StepLR(optimizer_res, step_size=args.decay_step_, gamma=args.decay_rate_)
            optimizer = {'org': optimizer_org, 'res': optimizer_res}
            scheduler = {'org': scheduler_org, 'res': scheduler_res}

            print(f"==================== After Removing the Unknown Concept ====================")
            acc_removed = eval(args, model, feature_extractor, test_loader, pred_type='res')

            model, metrics = discover(args, train_loader, test_loader, feature_extractor, model, criterion, optimizer, scheduler, candidate_concept_bank)
            discovered_cpt_name = discovery_recorder(args, model, metrics, cpt_num+1, acc_removed)
            pt_mdl, concept_bank, candidate_concept_bank, unknown_cpts = update_concept_bank(model, concept_bank, discovered_cpt_name, candidate_concept_bank, unknown_cpts)


if __name__ == '__main__':
    main()