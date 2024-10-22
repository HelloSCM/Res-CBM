# Incremental Residual Concept Bottleneck Models

This is the code repository of "Incremental Residual Concept Bottleneck Models".
The paper is accepted by [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Shang_Incremental_Residual_Concept_Bottleneck_Models_CVPR_2024_paper.html).
![image](https://github.com/user-attachments/assets/15d6b5aa-42ff-45ea-8432-43ea5336ad6e)

## Setup
- Setup environments:
```
pip install -r requirements.txt
```

- Prepare concept bank:
```
# ConceptNet concepts:
python conceptnet_concepts.py --save_path ...

# Visual Genome concepts (candidate concept bank):
run generate_vg_concepts.ipynb

# Res-CBM concepts (base concept bank):
run generate_res_concepts.ipynb
```

- Get image representations:
```
python get_img_repre.py
```

- Prepare datasets:

You can implement your ```Dataset``` and ```Dataloader``` in dataset.

## Run
- First, you can run Residual Concept Bottleneck Models by command:
```
bash scripts/predict.sh

# or

python train.py --task prediction \
                --dataset cifar10 \
                --res_dim 10 \
                --cpt_path cifar10/cifar10_res_num_237_len_7.pkl
```

- Then, you can run Incremental Concept Discovery by command:
```
bash scripts/discover.sh

# or

python train.py --task discovery \
                --dataset cifar10 \
                --cpt_path concept_bank/cifar10/cifar10_res_num_237_len_7.pkl \
                --mdl_path results/prediction/cifar10/Cpt_237_Res_1_Acc_0.8785/model.pt \
                --candidate_path concept_bank/vg/vg_num_13936_len_9.pkl \
                --epochs 20 --init_lr 0.001 --decay_step 3 --decay_rate 0.5 --init_lr_ 0.01 --decay_step_ 3 --decay_rate_ 0.5 \
                --res_dim 10 --sim_reg 0.1 --candidate_num 5
```
