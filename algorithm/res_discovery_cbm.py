import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResDiscoveryCBM(nn.Module):
    def __init__(self, class_num, concept_bank, unknown_concepts, reg_l1=0.0, reg_sim=10.0):
        self.concept_bank = torch.from_numpy(np.array(list(concept_bank.values()))).float()
        super(ResDiscoveryCBM, self).__init__()
        self.reg_l1 = reg_l1
        self.reg_sim = reg_sim

        self.original_classifier = nn.Linear(len(concept_bank) + 1, class_num)
        if unknown_concepts.shape[0] > 1:
            self.residual_classifier = nn.Linear(unknown_concepts.shape[0] - 1, class_num)

        mean_concept = torch.mean(self.concept_bank, dim=0).view(1, -1)
        noise = torch.randn_like(mean_concept)
        self.discovered_concept = nn.Parameter(mean_concept + 0.1 * noise)
        self.unknown_concepts = unknown_concepts[:-1]


    def discover(self, feas, labels, criterion, optimizer, candidate_concepts, candidate_num):
        if self.unknown_concepts.shape[0] >= 1:
            for param in self.original_classifier.parameters():
                param.requires_grad = True
            for param in self.residual_classifier.parameters():
                param.requires_grad = False
            preds = self.predict(feas, pred_type='org')
            loss_c = criterion(preds, labels)

            sims = self.calculate_similarity(self.discovered_concept, candidate_concepts)
            max_sim, _ = torch.sort(torch.stack(list(sims.values())), descending=True)
            loss_c += torch.sum(1 - max_sim[:candidate_num]) * self.reg_sim

            l1 = 0.0
            for param in self.original_classifier.parameters():
                l1 += torch.norm(param, 1)
            loss_c += self.reg_l1 * l1

            optimizer['org'].zero_grad()
            loss_c.backward()
            optimizer['org'].step()

            for param in self.original_classifier.parameters():
                param.requires_grad = False
            for param in self.residual_classifier.parameters():
                param.requires_grad = True
            preds = self.predict(feas, pred_type='res')
            loss_r = criterion(preds, labels)

            l1 = 0.0
            for param in self.residual_classifier.parameters():
                l1 += torch.norm(param, 1)
            loss_r += self.reg_l1 * l1

            optimizer['res'].zero_grad()
            loss_r.backward()
            optimizer['res'].step()
        
        elif self.unknown_concepts.shape[0] == 0:
            for param in self.original_classifier.parameters():
                param.requires_grad = True
            preds = self.predict(feas, pred_type='org')
            loss_c = criterion(preds, labels)

            sims = self.calculate_similarity(self.discovered_concept, candidate_concepts)
            max_sim, _ = torch.sort(torch.stack(list(sims.values())), descending=True)
            loss_c += torch.sum(1 - max_sim[:candidate_num]) * self.reg_sim
            
            l1 = 0.0
            for param in self.original_classifier.parameters():
                l1 += torch.norm(param, 1)
            loss_c += self.reg_l1 * l1

            optimizer['org'].zero_grad()
            loss_c.backward()
            optimizer['org'].step()

        return loss_c.item()
        
    
    def calculate_similarity(self, discovered_concept, candidate_concepts):
        sims = {}
        for cpt_name, candidate_cpt in candidate_concepts.items():
            candidate_concept= torch.from_numpy(candidate_cpt).cuda()
            sim = F.cosine_similarity(discovered_concept.squeeze(), candidate_concept, dim=0)
            sims[cpt_name] = sim
            
        return sims


    def predict(self, fea, pred_type='res'):
        fea_img = nn.functional.normalize(fea, p=2, dim=1)
        fea_org_cpts = self.concept_bank.cuda()
        org_cpts = torch.matmul(fea_img, fea_org_cpts.T)

        if self.unknown_concepts.shape[0] <= 0 or pred_type == 'org':
            fea_disc_cpts = nn.functional.normalize(self.discovered_concept, p=2, dim=1).cuda()
            disc_cpts = torch.matmul(fea_img, fea_disc_cpts.T)
            known_cpts = torch.cat((org_cpts, disc_cpts), dim=1)

            normalized_known_cpts = (known_cpts - torch.mean(known_cpts, dim=1, keepdim=True)) / torch.std(known_cpts, dim=1, keepdim=True)
            out = self.original_classifier(normalized_known_cpts)
            return out
        
        elif pred_type == 'res':
            discovered_concept_copy = self.discovered_concept.detach()
            fea_disc_cpts = nn.functional.normalize(discovered_concept_copy, p=2, dim=1).cuda()
            disc_cpts = torch.matmul(fea_img, fea_disc_cpts.T)
            known_cpts = torch.cat((org_cpts, disc_cpts), dim=1)

            fea_unknown_cpts = nn.functional.normalize(self.unknown_concepts, p=2, dim=1).cuda()
            unknown_cpts = torch.matmul(fea_img, fea_unknown_cpts.T)
            all_cpts = torch.cat((known_cpts, unknown_cpts), dim=1)
            
            normalized_known_cpts = (known_cpts - torch.mean(all_cpts, dim=1, keepdim=True)) / torch.std(all_cpts, dim=1, keepdim=True)
            normalized_unknown_cpts = (unknown_cpts - torch.mean(all_cpts, dim=1, keepdim=True)) / torch.std(all_cpts, dim=1, keepdim=True)
            out = self.original_classifier(normalized_known_cpts)
            res = self.residual_classifier(normalized_unknown_cpts)
            return out + res