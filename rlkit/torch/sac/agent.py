import numpy as np

import torch
import copy
from torch import nn as nn
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu

from rlkit.torch.New_networks import series_decomposition


def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


def _mean_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of mean of gaussians
    '''
    mu = torch.mean(mus, dim=0)
    sigma_squared = torch.mean(sigmas_squared, dim=0)
    return mu, sigma_squared


def _natural_to_canonical(n1, n2):
    ''' convert from natural to canonical gaussian parameters '''
    mu = -0.5 * n1 / n2
    sigma_squared = -0.5 * 1 / n2
    return mu, sigma_squared


def _canonical_to_natural(mu, sigma_squared):
    ''' convert from canonical to natural gaussian parameters '''
    n1 = mu / sigma_squared
    n2 = -0.5 * 1 / sigma_squared
    return n1, n2


class PEARLAgent(nn.Module):

    def __init__(self,
                 latent_dim,
                 context_encoder,
                 context_score_encoder,
                 policy,
                 **kwargs
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.context_encoder = context_encoder

        self.context_score_encoder = context_score_encoder

        self.policy = policy


        self.recurrent = kwargs['recurrent']
        self.use_ib = kwargs['use_information_bottleneck']
        self.sparse_rewards = kwargs['sparse_rewards']
        self.use_next_obs_in_context = kwargs['use_next_obs_in_context']

        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))

        self.register_buffer('z_scores', torch.zeros(1, latent_dim))

        self.W = nn.Parameter(torch.rand(self.latent_dim, self.latent_dim))

        self.scores = None
        self.scores_sigmoid = None
        self.reverse_z_means = None

        self.clear_z()

    def clear_z(self, num_tasks=1):
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        # reset distribution over z to the prior
        mu = ptu.zeros(num_tasks, self.latent_dim)
        if self.use_ib:
            var = ptu.ones(num_tasks, self.latent_dim)
        else:
            var = ptu.zeros(num_tasks, self.latent_dim)
        self.z_means = mu
        self.z_vars = var
        self.sample_z()
        self.context = None

        self.scores = None
        self.scores_sigmoid = None
        self.reverse_z_means = None

    def detach_z(self):
        ''' disable backprop through z '''
        self.z = self.z.detach()
        if self.recurrent:
            self.context_encoder.hidden = self.context_encoder.hidden.detach()

    def update_onlineadapt_max(self, score, context):
        if score > self.is_onlineadapt_max_score:
            self.is_onlineadapt_max_score = score
            self.is_onlineadapt_max_context = self.context
            self.is_onlineadapt_max_z = (self.z_means, self.z_vars)
            self.is_onlineadapt_max_z_sample = self.z
            for c in context:
                self.update_context(c)
            self.infer_posterior(self.context)
            self.is_onlineadapt_max_upd_context = self.context
        else:
            self.context = self.is_onlineadapt_max_context
            self.set_z(self.is_onlineadapt_max_z[0], self.is_onlineadapt_max_z[1])
    def fix_update_onlineadapt_max(self, score, context):
        if score > self.is_onlineadapt_max_score:
            self.is_onlineadapt_max_context = self.context
            self.is_onlineadapt_max_z = (self.z_means, self.z_vars)
            self.is_onlineadapt_max_z_sample = self.z
            for c in context:
                self.update_context(c)
            self.infer_posterior(self.context)
            self.is_onlineadapt_max_upd_context = self.context
        else:
            self.context = self.is_onlineadapt_max_context
            self.set_z(self.is_onlineadapt_max_z[0], self.is_onlineadapt_max_z[1])

    def set_z(self, means, vars):
        self.z_means = means
        self.z_vars = vars
        self.sample_z()

    def set_z_sample(self, z):
        self.z = z

    def set_onlineadapt_z_sample(self):
        self.z = self.is_onlineadapt_max_z_sample

    def set_onlineadapt_update_context(self):
        self.context = self.is_onlineadapt_max_upd_context
        self.infer_posterior(self.context)

    def clear_onlineadapt_max(self):
        self.is_onlineadapt_max_score = -1e8
        self.is_onlineadapt_max_context = None
        self.is_onlineadapt_max_upd_context = None
        self.is_onlineadapt_max_z = None
        self.is_onlineadapt_max_z_sample = None
        self.old_onlineadapt_max_score = -1e8

    def update_context(self, inputs):
        ''' append single transition to the current context '''
        o, a, r, no, d, info = inputs
        if self.sparse_rewards:
            r = info['sparse_reward']
        o = ptu.from_numpy(o[None, None, ...])
        a = ptu.from_numpy(a[None, None, ...])
        r=np.array(r)
        if len(r.shape) == 0:
            r = ptu.from_numpy(np.array([r])[None, None, ...])
        else:
            r = ptu.from_numpy(r[None, None, ...])
        no = ptu.from_numpy(no[None, None, ...])
        if self.use_next_obs_in_context:
            data = torch.cat([o, a, r, no], dim=2)
        else:
            data = torch.cat([o, a, r], dim=2)
        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1)
 
    def update_context_dict(self, batch_dict, env):
        ''' append context dictionary containing single/multiple transitions to the current context '''
        o = ptu.from_numpy(batch_dict['observations'][None, ...])
        a = ptu.from_numpy(batch_dict['actions'][None, ...])
        next_o = ptu.from_numpy(batch_dict['next_observations'][None, ...])
        if callable(getattr(env, "sparsify_rewards", None)) and self.sparse_rewards:
            r = batch_dict['rewards']
            sr = []
            for r_entry in r:
                sr.append(env.sparsify_rewards(r_entry))
            r = ptu.from_numpy(np.array(sr)[None, ...])
        else:
            r = ptu.from_numpy(batch_dict['rewards'][None, ...])
        if self.use_next_obs_in_context:
            data = torch.cat([o, a, r, next_o], dim=2)
        else:
            data = torch.cat([o, a, r], dim=2)
        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1)

    def compute_kl_div(self):
        ''' compute KL( q(z|c) || r(z) ) '''
        prior = torch.distributions.Normal(ptu.zeros(self.latent_dim), 0.05*ptu.ones(self.latent_dim))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    def infer_posterior(self, context, task_indices=None):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        params = self.context_encoder(context)
        params = params.view(context.size(0), -1, self.context_encoder.output_size)

        if task_indices is None:
            self.task_indices = np.zeros((context.size(0),))
        elif not hasattr(task_indices, '__iter__'):
            self.task_indices = np.array([task_indices])
        else:
            self.task_indices = np.array(task_indices)
        if self.use_ib:
            mu = params[..., :self.latent_dim]
            sigma_squared = F.softplus(params[..., self.latent_dim:])
            z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
            self.z_means = torch.stack([p[0] for p in z_params])
            self.z_vars = torch.stack([p[1] for p in z_params])
        else:

            params, self.z_means, self.scores, self.scores_sigmoid,  _, self.reverse_z_means = self.context_score_encoder(params)
            self.z_vars = torch.std(params, dim=1)

        self.sample_z()

    def only_infer_context_embeddings_TACO(self, context):
        ''' compute q(z|c) as a function of input context and sample new z from it'''

        params = self.context_encoder(context)
        params = params.view(context.size(0), -1, self.context_encoder.output_size)
        if self.use_ib:
            mu = params[..., :self.latent_dim]
            sigma_squared = F.softplus(params[..., self.latent_dim:])
            z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
            z_means = torch.stack([p[0] for p in z_params])
            z_vars = torch.stack([p[1] for p in z_params])

            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in
                          zip(torch.unbind(z_means), torch.unbind(z_vars))]
            z = [d.rsample() for d in posteriors]
            z_embedding = torch.stack(z)

            return z_embeddings
        else:
            z_embeddings = params

            return z_embeddings

    def only_infer_context_embeddings(self, context):
        ''' compute q(z|c) as a function of input context and sample new z from it'''

        params = self.context_encoder(context)
        params = params.view(context.size(0), -1, self.context_encoder.output_size)

        if self.use_ib:
            mu = params[..., :self.latent_dim]
            sigma_squared = F.softplus(params[..., self.latent_dim:])
            z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
            z_means = torch.stack([p[0] for p in z_params])
            z_vars = torch.stack([p[1] for p in z_params])

            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in
                          zip(torch.unbind(z_means), torch.unbind(z_vars))]
            z = [d.rsample() for d in posteriors]
            z_embedding = torch.stack(z)

            return z_embeddings
        else:

            _, z_embeddings, _, _, _, _ = self.context_score_encoder(
                params)

            return z_embeddings

    def sample_z(self):
        if self.use_ib:
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
            z = [d.rsample() for d in posteriors]
            self.z = torch.stack(z)
        else:
            self.z = self.z_means


    def get_action(self, obs, deterministic=False):
        ''' sample action from the policy, conditioned on the task embedding '''
        z = self.z
        obs = ptu.from_numpy(obs[None])
        in_ = torch.cat([obs, z], dim=1)
        return self.policy.get_action(in_, deterministic=deterministic)

    def set_num_steps_total(self, n):
        self.policy.set_num_steps_total(n)

    def forward(self, obs, context, task_indices=None, for_update=False):
        ''' given context, get statistics under the current policy of a set of observations '''
        self.infer_posterior(context, task_indices=task_indices)
        self.sample_z()

        task_z = self.z
        reverse_task_z = self.reverse_z_means

        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        task_z = [z.repeat(b, 1) for z in task_z]
        task_z = torch.cat(task_z, dim=0)
        in_ = torch.cat([obs, task_z], dim=1)
        policy_outputs = self.policy(t, b, in_, reparameterize=True, return_log_prob=True)

        reverse_task_z = [z.repeat(b, 1) for z in reverse_task_z]
        reverse_task_z = torch.cat(reverse_task_z, dim=0)

        if not for_update:
            if not self.use_ib:
                task_z_vars = [z.repeat(b, 1) for z in self.z_vars]
                task_z_vars = torch.cat(task_z_vars, dim=0)
                return policy_outputs, task_z, task_z_vars
            return policy_outputs, task_z
        else:
            if not self.use_ib:
                task_z_vars = [z.repeat(b, 1) for z in self.z_vars]
                task_z_vars = torch.cat(task_z_vars, dim=0)
                return policy_outputs, task_z, task_z_vars, self.scores, self.scores_sigmoid, reverse_task_z
            return policy_outputs, task_z, self.scores, self.scores_sigmoid, reverse_task_z

    def log_diagnostics(self, eval_statistics):


        for i in range(len(self.z_means[0])):
            z_mean = ptu.get_numpy(self.z_means[0][i])
            name = 'Z mean eval' + str(i)
            eval_statistics[name] = z_mean

        print("z_vars: ", self.z_vars)
        print("z_vars.shape: ", self.z_vars.shape)
        z_sig = np.mean(ptu.get_numpy(self.z_vars[0]))
        eval_statistics['Z variance eval'] = z_sig


        eval_statistics['task_idx'] = self.task_indices[0]

    @property
    def networks(self):
        return [self.context_encoder, self.policy]


class OldPEARLAgent(nn.Module):

    def __init__(self,
                 latent_dim,
                 context_encoder,
                 policy,
                 **kwargs
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.context_encoder = context_encoder
        self.policy = policy

        self.recurrent = kwargs['recurrent']
        self.use_ib = kwargs['use_information_bottleneck']
        self.sparse_rewards = kwargs['sparse_rewards']
        self.use_next_obs_in_context = kwargs['use_next_obs_in_context']

        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))

        self.clear_z()

    def clear_z(self, num_tasks=1):
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        mu = ptu.zeros(num_tasks, self.latent_dim)
        if self.use_ib:
            var = ptu.ones(num_tasks, self.latent_dim)
        else:
            var = ptu.zeros(num_tasks, self.latent_dim)
        self.z_means = mu
        self.z_vars = var
        self.sample_z()
        self.context = None

    def detach_z(self):
        ''' disable backprop through z '''
        self.z = self.z.detach()
        if self.recurrent:
            self.context_encoder.hidden = self.context_encoder.hidden.detach()

    def update_onlineadapt_max(self, score, context):
        if score > self.is_onlineadapt_max_score:
            self.is_onlineadapt_max_score = score
            self.is_onlineadapt_max_context = self.context
            self.is_onlineadapt_max_z = (self.z_means, self.z_vars)
            self.is_onlineadapt_max_z_sample = self.z
            for c in context:
                self.update_context(c)
            self.infer_posterior(self.context)
            self.is_onlineadapt_max_upd_context = self.context
        else:
            self.context = self.is_onlineadapt_max_context
            self.set_z(self.is_onlineadapt_max_z[0], self.is_onlineadapt_max_z[1])

    def set_z(self, means, vars):
        self.z_means = means
        self.z_vars = vars
        self.sample_z()

    def set_z_sample(self, z):
        self.z = z

    def set_onlineadapt_z_sample(self):
        self.z = self.is_onlineadapt_max_z_sample

    def set_onlineadapt_update_context(self):
        self.context = self.is_onlineadapt_max_upd_context
        self.infer_posterior(self.context)

    def clear_onlineadapt_max(self):
        self.is_onlineadapt_max_score = -1e8
        self.is_onlineadapt_max_context = None
        self.is_onlineadapt_max_upd_context = None
        self.is_onlineadapt_max_z = None
        self.is_onlineadapt_max_z_sample = None
        self.old_onlineadapt_max_score =  -1e8

    def update_context(self, inputs):
        ''' append single transition to the current context '''
        o, a, r, no, d, info = inputs
        if self.sparse_rewards:
            r = info['sparse_reward']
        o = ptu.from_numpy(o[None, None, ...])
        a = ptu.from_numpy(a[None, None, ...])
        r=np.array(r)
        if len(r.shape) == 0:
            r = ptu.from_numpy(np.array([r])[None, None, ...])
        else:
            r = ptu.from_numpy(r[None, None, ...])
        no = ptu.from_numpy(no[None, None, ...])

        if self.use_next_obs_in_context:
            data = torch.cat([o, a, r, no], dim=2)
        else:
            data = torch.cat([o, a, r], dim=2)
        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1)

    def compute_kl_div(self):
        ''' compute KL( q(z|c) || r(z) ) '''
        prior = torch.distributions.Normal(ptu.zeros(self.latent_dim), ptu.ones(self.latent_dim))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    def infer_posterior(self, context, task_indices=None):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        params = self.context_encoder(context)
        params = params.view(context.size(0), -1, self.context_encoder.output_size)
        if task_indices is None:
            self.task_indices = np.zeros((context.size(0),))
        elif not hasattr(task_indices, '__iter__'):
            self.task_indices = np.array([task_indices])
        else:
            self.task_indices = np.array(task_indices)
        if self.use_ib:
            mu = params[..., :self.latent_dim]
            sigma_squared = F.softplus(params[..., self.latent_dim:])
            z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
            self.z_means = torch.stack([p[0] for p in z_params])
            self.z_vars = torch.stack([p[1] for p in z_params])
        else:
            self.z_means = torch.mean(params, dim=1)
        self.sample_z()

    def sample_z(self):
        if self.use_ib:
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
            z = [d.rsample() for d in posteriors]
            self.z = torch.stack(z)
        else:
            self.z = self.z_means

    def get_action(self, obs, deterministic=False):
        ''' sample action from the policy, conditioned on the task embedding '''
        z = self.z
        obs = ptu.from_numpy(obs[None])
        in_ = torch.cat([obs, z], dim=1)
        return self.policy.get_action(in_, deterministic=deterministic)

    def set_num_steps_total(self, n):
        self.policy.set_num_steps_total(n)

    def forward(self, obs, context, task_indices=None):
        ''' given context, get statistics under the current policy of a set of observations '''
        self.infer_posterior(context, task_indices=task_indices)
        self.sample_z()

        task_z = self.z

        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        task_z = [z.repeat(b, 1) for z in task_z]
        task_z = torch.cat(task_z, dim=0)
        in_ = torch.cat([obs, task_z.detach()], dim=1) # in focal these does not use detach()
        policy_outputs = self.policy(t, b, in_, reparameterize=True, return_log_prob=True)

        return policy_outputs, task_z

    def log_diagnostics(self, eval_statistics):
        '''
        adds logging data about encodings to eval_statistics
        '''
        z_mean = np.mean(np.abs(ptu.get_numpy(self.z_means[0])))
        z_sig = np.mean(ptu.get_numpy(self.z_vars[0]))
        eval_statistics['Z mean eval'] = z_mean
        eval_statistics['Z variance eval'] = z_sig

    @property
    def networks(self):
        return [self.context_encoder, self.policy]
