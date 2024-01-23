#!/bin/bash
   # Script to reproduce results

 Foldername="0723_offline_meta_rl_reccc"
 mkdir out_logs/${Foldername} &> /dev/null
 declare -a tasks=( "cpearl-sparse-point-robot" )
 #  "cpearl-sparse-point-robot"
 declare -a algos=( "cpearl" )
 ##
 declare -a seeds=( "26" )
 declare -a datadirs=( "sparse-point-robot" )
 # "sparse-point-robot-20"
 declare -a is_sparses=( "0" )
 declare -a use_bracs=( "0" )
 declare -a use_information_bottlenecks=( "0" )
 declare -a is_zlosses=( "1" )
 declare -a is_onlineadapt_threses=( "0" )
 declare -a is_onlineadapt_maxes=( "1" )
 declare -a num_exp_traj_evals=(  "10" )
 declare -a allow_backward_zs=( "0" )
 declare -a is_true_sparses=( "0" )
 declare -a r_threses=( "-1000000" )
 declare -a wandb_project_names=( "" )
 declare -a wandb_run_names=( "" )
 declare -a csv_names=( "" )
 n=3
 # 52 54 21
 # 52 54 21 dense
 gpunum=4
 for task in "${tasks[@]}"
 do
 for algo in "${algos[@]}"
 do
 for seed in "${seeds[@]}"
 do
 for datadir in "${datadirs[@]}"
 do
 for is_sparse in "${is_sparses[@]}"
 do
 for use_brac in "${use_bracs[@]}"
 do
 for use_information_bottleneck in "${use_information_bottlenecks[@]}"
 do
 for is_zloss in "${is_zlosses[@]}"
 do
 for is_onlineadapt_thres in "${is_onlineadapt_threses[@]}"
 do
 for is_onlineadapt_max in "${is_onlineadapt_maxes[@]}"
 do
 for num_exp_traj_eval in "${num_exp_traj_evals[@]}"
 do
 for allow_backward_z in "${allow_backward_zs[@]}"
 do
 for is_true_sparse in "${is_true_sparses[@]}"
 do
 for r_thres in "${r_threses[@]}"
 do
 for wandb_project_name in "${wandb_project_names[@]}"
 do
 for wandb_run_name in "${wandb_run_names[@]}"
 do
 for csv_name in "${csv_names[@]}"
 do
 OMP_NUM_THREADS=16 KMP_AFFINITY="compact,granularity\=fine" nohup python launch_experiment_${algo}.py \
 ./configs/cpearl-sparse-point-robot-1.json \
 ./data/${datadir} \
 --gpu=${n} \
 --is_sparse_reward=${is_sparse} \
 --use_brac=${use_brac} \
 --use_information_bottleneck=${use_information_bottleneck} \
 --is_zloss=${is_zloss} \
 --is_onlineadapt_thres=${is_onlineadapt_thres} \
 --is_onlineadapt_max=${is_onlineadapt_max} \
 --num_exp_traj_eval=${num_exp_traj_eval} \
 --allow_backward_z=${allow_backward_z} \
 --is_true_sparse_rewards=${is_true_sparse} \
 --r_thres=${r_thres} \
 --wandb_project_name=${wandb_project_name} \
 --wandb_run_name=${wandb_run_name} \
 --csv_name=${csv_name} \
 >& out_logs/${Foldername}/${wandb_run_name}_${task}_${algo}_${datadir}_${is_sparse}_${use_brac}_${use_information_bottleneck}_${is_zloss}_${is_onlineadapt_thres}_${is_onlineadapt_max}_${num_exp_traj_eval}_${allow_backward_z}_${is_true_sparse}_${r_thres}_${seed}_${n}.txt &
 echo "task: ${task}, algo: ${algo}, datadir: ${datadir}, is_sparse: ${is_sparse}, use_brac: ${use_brac}"
 echo "     use_information_bottleneck: ${use_information_bottleneck}, is_zloss: ${is_zloss}"
 echo "     is_onlineadapt_thres: ${is_onlineadapt_thres}, is_onlineadapt_max: ${is_onlineadapt_max}"
 echo "     num_exp_traj_eval: ${num_exp_traj_eval}, allow_backward_z: ${allow_backward_z}"
 echo "     is_true_sparse: ${is_true_sparse}, r_thres: ${r_thres}, seed: ${seed}, GPU: $n"
 echo "     wandb_project_name: ${wandb_project_name}, wandb_run_name: ${wandb_run_name}, csv_name: ${csv_name}"
 n=$[($n+1) % ${gpunum}]
 sleep 10
 done
 done
 done
 done
 done
 done
 done
 done
 done
 done
 done
 done
 done
 done
 done
 done
 done
# CUDA_VISIBLE_DEVICES=3 python policy_train.py ./configs/sparse-point-robot.json
# python policy_train.py ./configs/sparse-point-robot.json --is_uniform