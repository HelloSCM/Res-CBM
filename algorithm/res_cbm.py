import torch
import torch.nn as nn
import numpy as np
        

class ResCBM(nn.Module):
    def __init__(self, class_num, res_dim, concept_bank, reg):
        self.concept_bank = torch.from_numpy(np.array(list(concept_bank.values()))).float()
        super(ResCBM, self).__init__()
        self.classifier = nn.Linear(len(concept_bank) + res_dim, class_num)
        self.original_classifier = nn.Linear(len(concept_bank), class_num)
        self.residual_classifier = nn.Linear(res_dim, class_num)
        self.reg = reg
        
        unknown_concepts = []
        for r in range(res_dim):
            tensor_rand = torch.randn(self.concept_bank.shape[1]).float()
            unknown_concepts.append(tensor_rand)
        self.unknown_concepts = nn.Parameter(torch.stack(unknown_concepts))
        
        
    def update(self, feas, labels, criterion, optimizer, mode='in-process', pred_type='org'):
        if mode == 'direct':
            preds = self.predict(feas, pred_type='all')
            loss = criterion(preds, labels)
            
            l1_reg = 0.0
            for param in self.classifier.parameters():
                l1_reg += torch.norm(param, 1)
            loss += self.reg * l1_reg
        
            optimizer['org'].zero_grad()
            loss.backward()
            optimizer['org'].step()
            
            return loss.item()
            
        elif mode == 'in-process':
            for param in self.original_classifier.parameters():
                param.requires_grad = True
            for param in self.residual_classifier.parameters():
                param.requires_grad = False
            preds = self.predict(feas, pred_type='org')
            loss_c = criterion(preds, labels)
            
            l1_reg = 0.0
            for param in self.original_classifier.parameters():
                l1_reg += torch.norm(param, 1)
            loss_c += self.reg * l1_reg
        
            optimizer['org'].zero_grad()
            loss_c.backward()
            optimizer['org'].step()
            
            for param in self.original_classifier.parameters():
                param.requires_grad = False
            for param in self.residual_classifier.parameters():
                param.requires_grad = True
            preds = self.predict(feas, pred_type='res')
            loss_r = criterion(preds, labels)
            
            l1_reg = 0.0
            for param in self.residual_classifier.parameters():
                l1_reg += torch.norm(param, 1)
            loss_r += self.reg * l1_reg
            
            optimizer['res'].zero_grad()
            loss_r.backward()
            optimizer['res'].step()
            
            return (loss_c.item() + loss_r.item()) / 2
        
        elif mode == 'post-hoc':
            if pred_type == 'org':
                for param in self.original_classifier.parameters():
                    param.requires_grad = True
                for param in self.residual_classifier.parameters():
                    param.requires_grad = False
                preds = self.predict(feas, pred_type='org')
                loss_c = criterion(preds, labels)
                
                l1_reg = 0.0
                for param in self.original_classifier.parameters():
                    l1_reg += torch.norm(param, 1)
                loss_c += self.reg * l1_reg
                
                optimizer['org'].zero_grad()
                loss_c.backward()
                optimizer['org'].step()
                
                return loss_c.item()
                
            elif pred_type == 'res':
                for param in self.original_classifier.parameters():
                    param.requires_grad = False
                for param in self.residual_classifier.parameters():
                    param.requires_grad = True
                preds = self.predict(feas, pred_type='res')
                loss_r = criterion(preds, labels)
                
                l1_reg = 0.0
                for param in self.residual_classifier.parameters():
                    l1_reg += torch.norm(param, 1)
                loss_r += self.reg * l1_reg
                
                optimizer['res'].zero_grad()
                loss_r.backward()
                optimizer['res'].step()
                
                return loss_r.item()
            
    def predict(self, fea, pred_type='res'):
        fea_img = nn.functional.normalize(fea, p=2, dim=1)
        fea_org_cpts = self.concept_bank.cuda()
        org_cpts = torch.matmul(fea_img, fea_org_cpts.T)
        fea_res_cpts = nn.functional.normalize(self.unknown_concepts, p=2, dim=1).cuda()
        res_cpts = torch.matmul(fea_img, fea_res_cpts.T)
        all_cpts = torch.cat((org_cpts, res_cpts), dim=1)
        
        if pred_type == 'all':
            normalized_all_cpts = (all_cpts - torch.mean(all_cpts, dim=1, keepdim=True)) / torch.std(all_cpts, dim=1, keepdim=True)
            out = self.classifier(normalized_all_cpts)
            return out
        
        elif pred_type == 'org':
            normalized_org_cpts = (org_cpts - torch.mean(org_cpts, dim=1, keepdim=True)) / torch.std(org_cpts, dim=1, keepdim=True)
            out = self.original_classifier(normalized_org_cpts)
            return out
        
        elif pred_type == 'res':
            normalized_org_cpts = (org_cpts - torch.mean(all_cpts, dim=1, keepdim=True)) / torch.std(all_cpts, dim=1, keepdim=True)
            normalized_res_cpts = (res_cpts - torch.mean(all_cpts, dim=1, keepdim=True)) / torch.std(all_cpts, dim=1, keepdim=True)
            out = self.original_classifier(normalized_org_cpts)
            res = self.residual_classifier(normalized_res_cpts)
            return out + res