import requests
import os
import pickle
import torch
import clip
import argparse
import numpy as np
from tqdm import tqdm


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", required=True, type=str)
    parser.add_argument("--dataset", default="tinyimagenet", type=str)
    parser.add_argument("--recurse", default=1, type=int, help="How many times to recurse on the conceptnet graph")
    return parser.parse_args()


def get_single_concept_data(cls_name):
    if cls_name in concept_cache:
        return concept_cache[cls_name]
    
    all_concepts = []
    
    # Has relations
    has_query = "https://api.conceptnet.io/query?node=/c/en/{}&rel=/r/HasA&start=/c/en/{}"
    obj = requests.get(has_query.format(cls_name, cls_name)).json()
    for edge in obj["edges"]:
        all_concepts.append(edge['end']['label'])
    
    # Made of relations
    madeof_query = "https://api.conceptnet.io/query?node=/c/en/{}&rel=/r/MadeOf&start=/c/en/{}"
    obj = requests.get(madeof_query.format(cls_name, cls_name)).json()
    for edge in obj["edges"]:
        all_concepts.append(edge['end']['label'])
    
    # Properties of things
    property_query = "https://api.conceptnet.io/query?node=/c/en/{}&rel=/r/HasProperty&start=/c/en/{}"
    obj = requests.get(property_query.format(cls_name, cls_name)).json()
    for edge in obj["edges"]:
        all_concepts.append(edge['end']['label'])
    
    # Categorization concepts
    is_query = "https://api.conceptnet.io/query?node=/c/en/{}&rel=/r/IsA&start=/c/en/{}"
    obj = requests.get(is_query.format(cls_name, cls_name)).json()
    for edge in obj["edges"]:
        if edge["weight"] <= 1:
            continue
        all_concepts.append(edge['end']['label'])
    
    # Parts of things
    parts_query = "https://api.conceptnet.io/query?node=/c/en/{}&rel=/r/PartOf&end=/c/en/{}"
    obj = requests.get(parts_query.format(cls_name, cls_name)).json()
    for edge in obj["edges"]:
        all_concepts.append(edge['start']['label'])
    
    all_concepts = [c.lower() for c in all_concepts]
    # Drop the "a " for concepts defined like "a {concept}".
    all_concepts = [c.replace("a ", "") for c in all_concepts]
    # Drop all empty concepts.
    all_concepts = [c for c in all_concepts if c!=""]
    # Make each concept unique in the set.
    all_concepts = set(all_concepts)
    
    concept_cache[cls_name] = all_concepts
    
    return all_concepts


def get_concept_data(all_classes):
    all_concepts = set()
    for cls_name in all_classes:
        print(f"Pulling concepts for {cls_name}")
        all_concepts |= get_single_concept_data(cls_name)
    return all_concepts


def clean_concepts(scenario_concepts):
    from nltk.stem.wordnet import WordNetLemmatizer
    import nltk

    nltk.download("wordnet")
    nltk.download("omw-1.4")
    Lem = WordNetLemmatizer()

    scenario_concepts_rec = []
    for c_prev in scenario_concepts:
        c = c_prev
        c = c.strip()
        c_subwords = c.split(" ")
        if len(c_subwords) > 2:
            print("skipping long concept", c_prev)
            continue
        for i, csw in enumerate(c_subwords):
            c_subwords[i] = Lem.lemmatize(csw)
        lemword = " ".join(c_subwords)
        if c_prev == lemword:
            scenario_concepts_rec.append(c)
        else:
            if lemword in scenario_concepts:
                print(c, lemword)
            else:
                scenario_concepts_rec.append(c)
    scenario_concepts_rec = list(set(scenario_concepts_rec))
    return scenario_concepts_rec


@torch.no_grad()
def learn_conceptbank(args, concept_list):
    concept_dict = {}
    for concept in tqdm(concept_list):
        text = clip.tokenize(f"{concept}").cuda()
        text_features = model.encode_text(text).cpu().numpy()
        text_features = text_features / np.linalg.norm(text_features)
        concept_dict[concept] = text_features

    print(f"# concepts: {len(concept_dict)}")
    concept_dict_path = os.path.join(args.save_path, f"{args.dataset}_pcbm_concepts_{len(concept_dict)}.pkl")
    pickle.dump(concept_dict, open(concept_dict_path, 'wb'))
    print(f"Dumped to : {concept_dict_path}")


if __name__ == "__main__":
    args = config()
    model, preprocess = clip.load('RN50', device='cuda')
    concept_cache = {}

    if args.dataset == 'cifar10':
        from torchvision import datasets
        cifar10 = datasets.CIFAR10(root='/data/.../', train=True, download=False)
        all_classes = list(cifar10.classes)
        
    elif args.dataset == 'cifar100':
        from torchvision import datasets
        cifar100 = datasets.CIFAR10(root='/data/.../', train=True, download=False)
        all_classes = list(cifar100.classes)

    all_concepts = get_concept_data(all_classes)
    all_concepts = clean_concepts(all_concepts)
    all_concepts = list(set(all_concepts).difference(set(all_classes)))
    for i in range(1, args.recurse):
        all_concepts = get_concept_data(all_concepts)
        all_concepts = list(set(all_concepts))
        all_concepts = clean_concepts(all_concepts)
        all_concepts = list(set(all_concepts).difference(set(all_classes)))
    learn_conceptbank(args, all_concepts)
