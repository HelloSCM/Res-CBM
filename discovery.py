import pickle
import torch
import torch.nn.functional as F
from train import eval


def update_init_weights(new_model, pt_model):
    new_model.original_classifier.weight.data[:, :-1] = pt_model.original_classifier.weight.data
    new_model.original_classifier.bias.data = pt_model.original_classifier.bias.data
    if new_model.unknown_concepts.shape[0] >= 1:
        new_model.residual_classifier.weight.data = pt_model.residual_classifier.weight.data[:, :-1]
        new_model.residual_classifier.bias.data = pt_model.residual_classifier.bias.data

    with open('concept_bank/cifar10/cifar10_class_num_10_len_6.pkl', 'rb') as cpt_file:
        class_concept = pickle.load(cpt_file)
    disc_cpt_weight = []
    for cls_cpt in class_concept.values():
        cls_cpt= torch.from_numpy(cls_cpt)
        sim = F.cosine_similarity(new_model.discovered_concept.squeeze(), cls_cpt, dim=0)
        disc_cpt_weight.append(sim)
    disc_cpt_weight = torch.stack(disc_cpt_weight)
    new_model.original_classifier.weight.data[:, -1] = disc_cpt_weight
        
    print("Reload Pretrained Model Weights...")
        
    return new_model


def update_concept_bank(model, concept_bank, discovered_cpt_name, candidate_concept_bank, unknown_cpts):
    discovered_cpt = candidate_concept_bank.pop(discovered_cpt_name)
    concept_bank[discovered_cpt_name] = discovered_cpt
    unknown_cpts = unknown_cpts[:-1]

    print("Update Concept Bank...")
        
    return model, concept_bank, candidate_concept_bank, unknown_cpts


def discover(args, train_loader, test_loader, feature_extractor, model, criterion, optimizer, scheduler, candidate_concept_bank):
    metrics = {'Loss':[], 'Acc_org': [], 'Acc_res': [], 'Concept': [], 'Sim': []}

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for batch_id, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.cuda(), labels.cuda()
            feas = feature_extractor(imgs)
            batch_result = model.discover(feas, labels, criterion, optimizer, candidate_concept_bank, args.candidate_num)
            running_loss += batch_result

        scheduler['org'].step()
        scheduler['res'].step()
            
        print(f"==================== Epoch [{epoch+1}/{args.epochs}] ====================")
        print(f"Train - Loss: {running_loss/batch_id:.4f}")
        
        print("Test Type: original concepts")
        acc_org = eval(args, model, feature_extractor, test_loader, pred_type='org')
        print("Test Type: residual concepts")
        acc_res = eval(args, model, feature_extractor, test_loader, pred_type='res')

        Sims = model.calculate_similarity(model.discovered_concept, candidate_concept_bank)
        cpt_idx = torch.argmax(torch.stack(list(Sims.values())))
        most_sim_cpt = list(Sims.keys())[cpt_idx]
        sim = Sims[most_sim_cpt].item()
        print(f"Most Similar Concept - {most_sim_cpt}: {sim:.4f}")
        
        metrics['Loss'].append(running_loss/batch_id)
        metrics['Acc_org'].append(acc_org)
        metrics['Acc_res'].append(acc_res)
        metrics['Concept'].append(most_sim_cpt)
        metrics['Sim'].append(sim)

    print("Finish Recovering!")
        
    return model, metrics