python train.py --task discovery \
                --dataset cifar10 \
                --cpt_path concept_bank/cifar10/cifar10_res_num_237_len_7.pkl \
                --mdl_path results/prediction/cifar10/Cpt_237_Res_1_Acc_0.8785/model.pt --res_dim 1 \
                --candidate_path concept_bank/vg/vg_num_13936_len_9.pkl \
                --epochs 20 --init_lr 0.001 --decay_step 3 --decay_rate 0.5 --init_lr_ 0.01 --decay_step_ 3 --decay_rate_ 0.5 \
                --sim_reg 0.1 --candidate_num 5