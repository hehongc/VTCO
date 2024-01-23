"""
Launcher for experiments with FOCAL

"""
import os
import pathlib
import numpy as np
import click
import json
import torch
import datetime
import multiprocessing as mp
from itertools import product
import sys
import gym

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder, Mlp
from rlkit.torch.sac.sac import CPEARL
from rlkit.torch.sac.agent import PEARLAgent, OldPEARLAgent
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs.default import default_config
from numpy.random import default_rng

import  metaworld,random,gym,gym.wrappers
from rlkit.envs.metaworld_wrapper import MetaWorldWrapper

import wandb
from rlkit.torch.New_networks import SelfAttnEncoder

rng = default_rng()

def global_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

def experiment(variant, seed=None):

    if 'v2' not in variant['env_name']:
        if 'normalizer.npz' in os.listdir(variant['algo_params']['data_dir']):
            obs_absmax = np.load(os.path.join(variant['algo_params']['data_dir'], 'normalizer.npz'))['abs_max']
            env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']), obs_absmax=obs_absmax)
        else:
            env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    else:
        if 'normalizer.npz' in os.listdir(variant['algo_params']['data_dir']):
            obs_absmax = np.load(os.path.join(variant['algo_params']['data_dir'], 'normalizer.npz'))['abs_max']
            ml1 = metaworld.ML1(variant['env_name'], seed=1337)  # Construct the benchmark, sampling tasks

            env = ml1.train_classes[variant['env_name']]()  # Create an environment with task
            env.train_tasks = ml1.train_tasks
            task = random.choice(ml1.train_tasks)
            env.set_task(task)

            tasks = list(range(len(env.train_tasks)))
            # env = gym.wrappers.TimeLimit(gym.wrappers.ClipAction(MetaWorldWrapper(env)), 500)
            env = gym.wrappers.TimeLimit(gym.wrappers.ClipAction(env), 500)
            env=MetaWorldWrapper(env)
            env.is_metaworld = 1
        else:
            ml1 = metaworld.ML1(variant['env_name'], seed=1337)  # Construct the benchmark, sampling tasks

            env = ml1.train_classes[variant['env_name']]()  # Create an environment with task
            # print(ml1.train_tasks)
            env.train_tasks = ml1.train_tasks
            task = random.choice(ml1.train_tasks)
            env.set_task(task)

            tasks = list(range(len(env.train_tasks)))
            # env = gym.wrappers.TimeLimit(gym.wrappers.ClipAction(MetaWorldWrapper(env)), 500)
            env = gym.wrappers.TimeLimit(gym.wrappers.ClipAction(env), 500)
            env = MetaWorldWrapper(env)
            env.is_metaworld = 1

    if seed is not None:
        global_seed(seed)
        env.seed(seed)

    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1
    task_dim = variant['env_params']['n_tasks']

    latent_dim = variant['latent_size']
    context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim if variant['algo_params']['use_next_obs_in_context'] else obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder

    context_encoder = encoder_model(
        hidden_sizes=[200, 200, 200],
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
        output_activation=torch.tanh,
    )

    context_score_encoder = SelfAttnEncoder(
        input_dim=context_encoder_output_dim,
        num_output_mlp=0,
    )

    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1,
    )

    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )

    if variant['algo_params']["is_zloss"] and not variant['algo_params']["use_information_bottleneck"]:
        agent = PEARLAgent(
            latent_dim,
            context_encoder,
            context_score_encoder,
            policy,
            **variant['algo_params']
        )

    else:
        agent = OldPEARLAgent(
            latent_dim,
            context_encoder,
            policy,
            **variant['algo_params']
        )

    rew_decoder = FlattenMlp(hidden_sizes=[net_size, net_size, net_size],
                             input_size=latent_dim  + obs_dim + action_dim,
                             output_size=1, )

    transition_decoder = FlattenMlp(hidden_sizes=[net_size, net_size, net_size],
                                    input_size=latent_dim  + obs_dim + action_dim,
                                    output_size=obs_dim, )

    task_id_decoder = FlattenMlp(hidden_sizes=[net_size],
                                 input_size=latent_dim,
                                 output_size=task_dim, )

    if variant['algo_type'] == 'CPEARL':
        # critic network for divergence in dual form (see BRAC paper https://arxiv.org/abs/1911.11361)
        c = FlattenMlp(
            hidden_sizes=[net_size, net_size, net_size],
            input_size=obs_dim + action_dim + latent_dim,
            output_size=1
        )
        if 1:  # foce random
            rng = default_rng()
            train_tasks = np.sort(rng.choice(len(tasks), size=variant['n_train_tasks'], replace=False))
            eval_tasks = set(range(len(tasks))).difference(train_tasks)
            if 'goal_radius' in variant['env_params']:
                algorithm = CPEARL(
                    env=env,
                    train_tasks=train_tasks,
                    eval_tasks=eval_tasks,
                    nets=[agent, qf1, qf2, vf, c, rew_decoder, transition_decoder, task_id_decoder],
                    latent_dim=latent_dim,
                    goal_radius=variant['env_params']['goal_radius'],
                    wandb_project_name=variant['wandb_project_name'],
                    wandb_run_name=variant['wandb_run_name'],
                    csv_name=variant['csv_name'],
                    **variant['algo_params']
                )
            else:
                algorithm = CPEARL(
                    env=env,
                    train_tasks=train_tasks,
                    eval_tasks=eval_tasks,
                    nets=[agent, qf1, qf2, vf, c,rew_decoder,transition_decoder, task_id_decoder],
                    latent_dim=latent_dim,
                    wandb_project_name=variant['wandb_project_name'],
                    wandb_run_name=variant['wandb_run_name'],
                    csv_name=variant['csv_name'],
                    **variant['algo_params']
                )
        else:
            if 'goal_radius' in variant['env_params']:
                algorithm = CPEARL(
                    env=env,
                    train_tasks=list(tasks[:variant['n_train_tasks']]),
                    eval_tasks=list(tasks[-variant['n_eval_tasks']:]),
                    nets=[agent, qf1, qf2, vf, c, rew_decoder, transition_decoder, task_id_decoder],
                    latent_dim=latent_dim,
                    goal_radius=variant['env_params']['goal_radius'],
                    wandb_project_name=variant['wandb_project_name'],
                    wandb_run_name=variant['wandb_run_name'],
                    csv_name=variant['csv_name'],
                    **variant['algo_params']
                )
            else:
                algorithm = CPEARL(
                    env=env,
                    train_tasks=list(tasks[:variant['n_train_tasks']]),
                    eval_tasks=list(tasks[-variant['n_eval_tasks']:]),
                    nets=[agent, qf1, qf2, vf, c, rew_decoder, transition_decoder, task_id_decoder],
                    latent_dim=latent_dim,
                    wandb_project_name=variant['wandb_project_name'],
                    wandb_run_name=variant['wandb_run_name'],
                    csv_name=variant['csv_name'],
                    **variant['algo_params']
                )
    else:
        NotImplemented

    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    if ptu.gpu_enabled():
        algorithm.to()

    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))

    unique_token = "{}__{}".format(variant['env_name'], datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    variant['util_params']['unique_token'] = unique_token
    variant['util_params']['base_log_dir'] = os.path.join(variant['util_params']['base_log_dir'], "{}").format(unique_token)

    # TODO support Docker
    exp_id = 'debug' if DEBUG else None
    experiment_log_dir = setup_logger(
        variant['env_name'],
        variant=variant,
        exp_id=exp_id,
        base_log_dir=variant['util_params']['base_log_dir'],
        seed=seed,
        snapshot_mode="all"
    )

    if variant['algo_params']['dump_eval_paths']:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    algorithm.train()

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

@click.command()
@click.argument('config', default=None)
@click.argument('data_dir', default=None)
@click.option('--gpu', default=3)
@click.option("--is_sparse_reward", default=0)
@click.option("--use_brac", default=0)
@click.option("--use_information_bottleneck", default=0)
@click.option("--is_zloss", default=0)
@click.option("--is_onlineadapt_thres", default=0)
@click.option("--is_onlineadapt_max", default=0)
@click.option("--num_exp_traj_eval", default=10)
@click.option("--allow_backward_z", default=0)
@click.option("--is_true_sparse_rewards", default=0)
@click.option("--r_thres", default=0.)
@click.option("--num_ensemble", default=4)
@click.option("--wandb_project_name", default='')
@click.option("--wandb_run_name", default='')
@click.option("--csv_name", default='')
def main(config, data_dir, gpu, is_sparse_reward, use_brac, use_information_bottleneck, is_zloss, is_onlineadapt_thres,
         is_onlineadapt_max, num_exp_traj_eval, allow_backward_z, is_true_sparse_rewards, r_thres, wandb_project_name, wandb_run_name, csv_name, num_ensemble):

    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['util_params']['gpu_id'] = gpu
    variant['algo_params']['data_dir'] = data_dir
    variant['algo_params']['sparse_rewards'] = is_sparse_reward
    variant['algo_params']['use_brac'] = use_brac
    variant['algo_params']['use_information_bottleneck'] = use_information_bottleneck
    variant['algo_params']['is_zloss'] = is_zloss
    variant['algo_params']['is_onlineadapt_thres'] = is_onlineadapt_thres
    variant['algo_params']['is_onlineadapt_max'] = is_onlineadapt_max
    variant['algo_params']['num_exp_traj_eval'] = num_exp_traj_eval
    variant['algo_params']['allow_backward_z'] = allow_backward_z
    variant['algo_params']['is_true_sparse_rewards'] = is_true_sparse_rewards
    variant['algo_params']['r_thres'] = r_thres
    variant['algo_params']['num_ensemble'] = num_ensemble

    variant['wandb_project_name'] = wandb_project_name
    variant['wandb_run_name'] = wandb_run_name
    variant['csv_name'] = csv_name

    p = mp.Pool(mp.cpu_count())
    if len(variant['seed_list']) > 0:
        p.starmap(experiment, product([variant], variant['seed_list']))
    else:
        experiment(variant)

if __name__ == "__main__":
    main()

