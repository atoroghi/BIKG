# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Optional, Callable
import math, sys
import logging

import torch
from torch import nn
from torch import optim
from torch import Tensor

from kbc.regularizers import Regularizer
import tqdm

import traceback

from kbc.utils import QuerDAG
from kbc.utils import DynKBCSingleton
from kbc.utils import make_batches
from kbc.utils import Device


class KBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def get_queries_separated(self, queries: torch.Tensor):
            pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    @abstractmethod
    def score_emb(self, lhs: torch.Tensor, rel: torch.Tensor, rhs: torch.Tensor):
        pass

    @abstractmethod
    def candidates_score(self, rel: Tensor, arg1: Optional[Tensor], arg2: Optional[Tensor], *args, **kwargs) -> Tuple[ Optional[Tensor ], Optional[Tensor]]:
        pass

    @abstractmethod
    def model_type(self):
        pass

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
        :param filters: filters[(lhs, rel)] gives the rhs to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size)
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    q = self.get_queries(these_queries)

                    scores = q @ rhs
                    targets = self.score(these_queries)

                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [int(x - c_begin) for x in filter_out if c_begin <= x < c_begin + chunk_size]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6

                    ranks[b_begin:b_begin + batch_size] += torch.sum((scores >= targets).float(), dim=1).cpu()

                    b_begin += batch_size

                c_begin += chunk_size

        return ranks

    @staticmethod
    def __get_chains__(chains: List, graph_type: str = QuerDAG.TYPE1_2.value):
        if graph_type == QuerDAG.TYPE1_1.value:
            chain1 = chains[0]
        elif '2' in graph_type[-1]:
            chain1, chain2 = chains
        elif '3' in graph_type[-1]:
            chain1, chain2, chain3 = chains

        if QuerDAG.TYPE1_1.value in graph_type:
            lhs_1 = chain1[0]
            rel_1 = chain1[1]

            raw_chain = [lhs_1, rel_1]

        elif QuerDAG.TYPE1_2.value in graph_type:
            lhs_1 = chain1[0]
            rel_1 = chain1[1]

            rel_2 = chain2[1]

            raw_chain = [lhs_1, rel_1, rel_2]

        elif QuerDAG.TYPE2_2.value in graph_type:
            lhs_1 = chain1[0]
            rel_1 = chain1[1]

            lhs_2 = chain2[0]
            rel_2 = chain2[1]

            raw_chain = [lhs_1, rel_1, lhs_2, rel_2]

        elif QuerDAG.TYPE1_3.value in graph_type:
            lhs_1 = chain1[0]
            rel_1 = chain1[1]

            rel_2 = chain2[1]

            rhs_3 = chain3[1]

            raw_chain = [lhs_1, rel_1, rel_2, rhs_3]

        elif QuerDAG.TYPE2_3.value in graph_type:
            lhs_1 = chain1[0]
            rel_1 = chain1[1]

            lhs_2 = chain2[0]
            rel_2 = chain2[1]

            lhs_3 = chain3[0]
            rel_3 = chain3[1]

            raw_chain = [lhs_1, rel_1, lhs_2, rel_2, lhs_3, rel_3]

        elif QuerDAG.TYPE3_3.value in graph_type:
            lhs_1 = chain1[0]
            rel_1 = chain1[1]

            rel_2 = chain2[1]

            lhs_2 = chain3[0]
            rel_3 = chain3[1]

            raw_chain = [lhs_1, rel_1, rel_2, lhs_2, rel_3]

        elif QuerDAG.TYPE4_3.value in graph_type:
            lhs_1 = chain1[0]
            rel_1 = chain1[1]

            lhs_2 = chain2[0]
            rel_2 = chain2[1]

            rel_3 = chain3[1]

            raw_chain = [lhs_1, rel_1, lhs_2, rel_2, rel_3]

        return raw_chain

    @staticmethod
    def _optimize_variables(scoring_fn: Callable, params: list, optimizer: str,
                            lr: float, max_steps: int):
        if optimizer == 'adam':
            optimizer = optim.Adam(params, lr=lr)
        elif optimizer == 'adagrad':
            optimizer = optim.Adagrad(params, lr=lr)
        elif optimizer == 'sgd':
            optimizer = optim.SGD(params, lr=lr)
        else:
            raise ValueError(f'Unknown optimizer {optimizer}')

        prev_loss_value = 1000
        loss_value = 999
        losses = []

        with tqdm.tqdm(total=max_steps, unit='iter', disable=False) as bar:
            i = 0
            while i < max_steps and math.fabs(prev_loss_value - loss_value) > 1e-9:
                prev_loss_value = loss_value

                norm, regularizer, _ = scoring_fn()
                loss = -norm.mean() + regularizer

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                i += 1
                bar.update(1)
                bar.set_postfix(loss=f'{loss.item():.6f}')

                loss_value = loss.item()
                losses.append(loss_value)

            if i != max_steps:
                bar.update(max_steps - i + 1)
                bar.close()
                print("Search converged early after {} iterations".format(i))

        with torch.no_grad():
            *_, scores = scoring_fn(score_all=True)

        return scores

    @staticmethod
    def batch_t_norm(atoms: Tensor, norm_type: str = 'min') -> Tensor:
        if norm_type == 'min':
            scores = torch.min(atoms, dim=-1)[0]
        elif norm_type == 'prod':
            scores = torch.prod(atoms, dim=-1)
        else:
            raise ValueError(f't_norm must be "min" or "prod", got {norm_type}')

        return scores

    @staticmethod
    def batch_t_conorm(atoms: Tensor, norm_type: str = 'max') -> Tensor:
        if norm_type == 'min':
            scores = torch.max(atoms, dim=-1)[0]
        elif norm_type == 'prod':
            scores = torch.sum(atoms, dim=-1) - torch.prod(atoms, dim=-1)
        else:
            raise ValueError(f't_conorm must be "min" or "prod", got {norm_type}')

        return scores

    def link_prediction(self, chains: List):
        lhs_1, rel_1 = self.__get_chains__(chains, graph_type=QuerDAG.TYPE1_1.value)
        # self.forward_emb returns the score of each node to be the rhs of the given lhs and rel
        return self.forward_emb(lhs_1, rel_1)

    def optimize_chains(self, chains: List, regularizer: Regularizer,
                        max_steps: int = 20, lr: float = 0.1,
                        optimizer: str = 'adam', norm_type: str = 'min'):
        def scoring_fn(score_all=False):
            # score_1: score of the proposed fact, factors_1: regularization loss of the nodes and rel of the 
            # proposed fact
            score_1, factors_1 = self.score_emb(lhs_1, rel_1, obj_guess_1)
            score_2, factors_2 = self.score_emb(obj_guess_1, rel_2, obj_guess_2)
            factors = [factors_1[2], factors_2[2]]

            atoms = torch.sigmoid(torch.cat((score_1, score_2), dim=1))

            if len(chains) == 3:
                score_3, factors_3 = self.score_emb(obj_guess_2, rel_3, obj_guess_3)
                factors.append(factors_3[2])
                atoms = torch.cat((atoms, torch.sigmoid(score_3)), dim=1)

            guess_regularizer = regularizer(factors)
            t_norm = self.batch_t_norm(atoms, norm_type)

            all_scores = None
            if score_all:
                if len(chains) == 2:
                    score_2 = self.forward_emb(obj_guess_1, rel_2)
                    atoms = torch.sigmoid(torch.stack((score_1.expand_as(score_2), score_2), dim=-1))
                else:
                    score_3 = self.forward_emb(obj_guess_2, rel_3)
                    atoms = torch.sigmoid(torch.stack((score_1.expand_as(score_3), score_2.expand_as(score_3), score_3), dim=-1))

                all_scores = self.batch_t_norm(atoms, norm_type)

            return t_norm, guess_regularizer, all_scores

        if len(chains) == 2:
            lhs_1, rel_1, rel_2 = self.__get_chains__(chains, graph_type=QuerDAG.TYPE1_2.value)
        elif len(chains) == 3:
            lhs_1, rel_1, rel_2, rel_3 = self.__get_chains__(chains, graph_type=QuerDAG.TYPE1_3.value)
        else:
            assert False, f'Invalid number of chains: {len(chains)}'

        obj_guess_1 = torch.normal(0, self.init_size, lhs_1.shape, device=lhs_1.device, requires_grad=True)
        obj_guess_2 = torch.normal(0, self.init_size, lhs_1.shape, device=lhs_1.device, requires_grad=True)
        params = [obj_guess_1, obj_guess_2]
        if len(chains) == 3:
            obj_guess_3 = torch.normal(0, self.init_size, lhs_1.shape, device=lhs_1.device, requires_grad=True)
            params.append(obj_guess_3)

        scores = self._optimize_variables(scoring_fn, params, optimizer, lr, max_steps)
        return scores

    def optimize_chains_bpl(self, chains: List, regularizer: Regularizer,
                        cov_anchor: float = 0.1,
                        cov_var: float = 0.1, cov_target: float = 0.1, possible_heads_emb: list = None, possible_tails_emb: list = None,
                        all_nodes_embs: torch.tensor = None):

        
        if len(chains) == 2:
            emb_dim = chains[0][0].shape[1] // 2
            all_heads_embs = all_nodes_embs[:, :emb_dim]
            all_tails_embs = all_nodes_embs[:, emb_dim:]
            mu_m_for = possible_tails_emb[0][:, :emb_dim]
            h_m_for = (1/cov_var) * mu_m_for
            mu_m_inv = possible_tails_emb[0][:, emb_dim:]
            h_m_inv = (1/cov_var) * mu_m_inv
            #lhs_1 is a tensor of size (query_size, (2*)emb_dim)
            lhs_1, rel_1, rel_2 = self.__get_chains__(chains, graph_type=QuerDAG.TYPE1_2.value)
            # update the precision and information of the variable given the anchor
            # when updating m given d, we should use the tail embedding of d
            mu_d_for = lhs_1[:, emb_dim:] * rel_1[:, :emb_dim]
            #mu_d_for = lhs_1[:, :emb_dim] * rel_1[:, :emb_dim]
            h_d_for = (1/cov_anchor) * mu_d_for
            h_m_for = h_m_for + h_d_for
            J_m_for = (1/cov_anchor) + (1/cov_var)
            mu_d_inv = lhs_1[:, :emb_dim] * rel_1[:, emb_dim:]
            #mu_d_inv = lhs_1[:, emb_dim:] * rel_1[:, emb_dim:]
            h_d_inv = (1/cov_anchor) * mu_d_inv
            h_m_inv = h_m_inv + h_d_inv
            J_m_inv = (1/cov_anchor) + (1/cov_var)

            # update the precision and information of the target node given the variable
            mu_u_for = possible_tails_emb[1][:,:emb_dim]
            h_u_for = (1/cov_target) * mu_u_for
            mu_u_inv = possible_tails_emb[1][:,emb_dim:]
            h_u_inv = (1/cov_target) * mu_u_inv

            #TODO: check if this direction is correct or we should use the tail embedding of m
            #h_u_for = h_u_for - rel_2[:, :emb_dim] * (1 / J_m_for) * h_m_for
            h_u_for = h_u_for - rel_2[:, :emb_dim] * (1 / J_m_for) * h_m_inv
            
            J_u_for = (1/cov_target) - rel_2[:, :emb_dim] * (1 / J_m_for) * rel_2[:, :emb_dim]

            #h_u_inv = h_u_inv - rel_2[:, emb_dim:] * (1 / J_m_inv) * h_m_inv
            h_u_inv = h_u_inv - rel_2[:, emb_dim:] * (1 / J_m_inv) * h_m_for
            
            J_u_inv = (1/cov_target) - rel_2[:, emb_dim:] * (1 / J_m_inv) * rel_2[:, emb_dim:]

            mu_u_for = h_u_for / J_u_for
            mu_u_inv = h_u_inv / J_u_inv

        elif len(chains) == 3:
            emb_dim = chains[0][0].shape[1] // 2
            all_heads_embs = all_nodes_embs[:, :emb_dim]
            all_tails_embs = all_nodes_embs[:, emb_dim:]
            mu_m1_for = possible_tails_emb[0][:, :emb_dim]
            h_m1_for = (1/cov_var) * mu_m1_for
            mu_m1_inv = possible_tails_emb[0][:, emb_dim:]
            h_m1_inv = (1/cov_var) * mu_m1_inv
            #lhs_1 is a tensor of size (query_size, (2*)emb_dim)
            lhs_1, rel_1, rel_2, rel_3 = self.__get_chains__(chains, graph_type=QuerDAG.TYPE1_3.value)
            # update the precision and information of the first variable given the anchor
            mu_d_for = lhs_1[:, emb_dim:] * rel_1[:, :emb_dim]
            h_d_for = (1/cov_anchor) * mu_d_for
            h_m1_for = h_m1_for + h_d_for
            J_m1_for = (1/cov_anchor) + (1/cov_var)
            mu_d_inv = lhs_1[:, :emb_dim] * rel_1[:, emb_dim:]
            h_d_inv = (1/cov_anchor) * mu_d_inv
            h_m1_inv = h_m1_inv + h_d_inv
            J_m1_inv = (1/cov_anchor) + (1/cov_var)

            # update the precision and information of the second variable given the first variable
            mu_m2_for = possible_tails_emb[1][:,:emb_dim]
            h_m2_for = (1/cov_var) * mu_m2_for
            mu_m2_inv = possible_tails_emb[1][:,emb_dim:]
            h_m2_inv = (1/cov_var) * mu_m2_inv
            h_m2_for = h_m2_for - rel_2[:, :emb_dim] * (1 / J_m1_for) * h_m1_for
            J_m2_for = (1/cov_var) - rel_2[:, :emb_dim] * (1 / J_m1_for) * rel_2[:, :emb_dim]
            h_m2_inv = h_m2_inv - rel_2[:, emb_dim:] * (1 / J_m1_inv) * h_m1_inv
            J_m2_inv = (1/cov_var) - rel_2[:, emb_dim:] * (1 / J_m1_inv) * rel_2[:, emb_dim:]

            # update the precision and information of the target node given the second variable
            mu_u_for = possible_tails_emb[2][:,:emb_dim]
            h_u_for = (1/cov_target) * mu_u_for
            mu_u_inv = possible_tails_emb[2][:,emb_dim:]
            h_u_inv = (1/cov_target) * mu_u_inv

            h_u_for = h_u_for - rel_3[:, :emb_dim] * (1 / J_m2_for) * h_m2_for
            J_u_for = (1/cov_target) - rel_3[:, :emb_dim] * (1 / J_m2_for) * rel_3[:, :emb_dim]

            h_u_inv = h_u_inv - rel_3[:, emb_dim:] * (1 / J_m2_inv) * h_m2_inv
            J_u_inv = (1/cov_target) - rel_3[:, emb_dim:] * (1 / J_m2_inv) * rel_3[:, emb_dim:]

            mu_u_for = h_u_for / J_u_for
            mu_u_inv = h_u_inv / J_u_inv
        
        else:
            assert False, f'Invalid number of chains: {len(chains)}'
        
        scores = []

        #a = torch.tensor ([1,2,3, -1])
        #b = torch.tensor ([[4,5,6,-2], [7,8,9,-2], [10,11,12,-2]])
        #c =0.5*( b@ a.T)
        #print(c)

        scores = torch.zeros( mu_u_for.shape[0], all_heads_embs.shape[0])

        for i in range(scores.shape[0]):
        #    # get the dot product between row[i] of mu_u_for and each row of all_heads_embs
            scores [i] = torch.clip( 0.5*( all_heads_embs @ (mu_u_for[i]).T + all_tails_embs @ (mu_u_inv[i]).T), min=-20, max=20)


        #obj_guess_1 = torch.normal(0, self.init_size, lhs_1.shape, device=lhs_1.device, requires_grad=True)
        #obj_guess_2 = torch.normal(0, self.init_size, lhs_1.shape, device=lhs_1.device, requires_grad=True)
        #params = [obj_guess_1, obj_guess_2]
        #if len(chains) == 3:
        #    obj_guess_3 = torch.normal(0, self.init_size, lhs_1.shape, device=lhs_1.device, requires_grad=True)
        #    params.append(obj_guess_3)

        #scores = self._optimize_variables(scoring_fn, params, optimizer, lr, max_steps)

        return scores

    def optimize_intersections_bpl(self, chains: List, regularizer: Regularizer,
                               max_steps: int = 20, lr: float = 0.1,
                               optimizer:str = 'adam', norm_type: str = 'min',
                               disjunctive=False, cov_anchor: float = 0.1,
                        cov_var: float = 0.1, cov_target: float = 0.1, possible_heads_emb: list = None, possible_tails_emb: list = None,
                        all_nodes_embs: torch.tensor = None):

        if len(chains) == 2:
            emb_dim = chains[0][0].shape[1] // 2
            all_heads_embs = all_nodes_embs[:, :emb_dim]
            all_tails_embs = all_nodes_embs[:, emb_dim:]
            mu_u_for = possible_tails_emb[0][:, :emb_dim]
            h_u_for = (1/cov_target) * mu_u_for
            mu_u_inv = possible_tails_emb[0][:, emb_dim:]
            h_u_inv = (1/cov_target) * mu_u_inv
            #lhs_1 is a tensor of size (query_size, (2*)emb_dim)
            raw_chain = self.__get_chains__(chains, graph_type=QuerDAG.TYPE2_2.value)
            lhs_1, rel_1, lhs_2, rel_2 = raw_chain
            mu_d1_for = lhs_1[:, emb_dim:] * rel_1[:, :emb_dim]
            h_d1_for = (1/cov_anchor) * mu_d1_for
            mu_d2_for = lhs_2[:, emb_dim:] * rel_2[:, :emb_dim]
            h_d2_for = (1/cov_anchor) * mu_d2_for
            mu_d1_inv = lhs_1[:, :emb_dim] * rel_1[:, emb_dim:]
            h_d1_inv = (1/cov_anchor) * mu_d1_inv
            mu_d2_inv = lhs_2[:, :emb_dim] * rel_2[:, emb_dim:]
            h_d2_inv = (1/cov_anchor) * mu_d2_inv

            if not disjunctive:
                h_u_for = h_u_for + h_d1_for + h_d2_for
                J_u_for = (1/cov_anchor) + (1/cov_anchor) + (1/cov_target)
                mu_u_for = h_u_for / J_u_for
                h_u_inv = h_u_inv + h_d1_inv + h_d2_inv
                J_u_inv = (1/cov_anchor) + (1/cov_anchor) + (1/cov_target)
                mu_u_inv = h_u_inv / J_u_inv
            elif disjunctive:
                h_u_for1 = h_u_for + h_d1_for
                J_u_for1 = (1/cov_anchor) + (1/cov_target)
                mu_u_for1 = h_u_for1 / J_u_for1
                h_u_for2 = h_u_for + h_d2_for
                J_u_for2 = (1/cov_anchor) + (1/cov_target)
                mu_u_for2 = h_u_for2 / J_u_for2
                h_u_inv1 = h_u_inv + h_d1_inv
                J_u_inv1 = (1/cov_anchor) + (1/cov_target)
                mu_u_inv1 = h_u_inv1 / J_u_inv1
                h_u_inv2 = h_u_inv + h_d2_inv
                J_u_inv2 = (1/cov_anchor) + (1/cov_target)
                mu_u_inv2 = h_u_inv2 / J_u_inv2



        elif len(chains) == 3:
            emb_dim = chains[0][0].shape[1] // 2
            all_heads_embs = all_nodes_embs[:, :emb_dim]
            all_tails_embs = all_nodes_embs[:, emb_dim:]
            mu_u_for = possible_tails_emb[0][:, :emb_dim]
            h_u_for = (1/cov_target) * mu_u_for
            mu_u_inv = possible_tails_emb[0][:, emb_dim:]
            h_u_inv = (1/cov_target) * mu_u_inv
            #lhs_1 is a tensor of size (query_size, (2*)emb_dim)
            raw_chain = self.__get_chains__(chains, graph_type=QuerDAG.TYPE2_3.value)
            lhs_1, rel_1, lhs_2, rel_2, lhs_3, rel_3 = raw_chain

            mu_d1_for = lhs_1[:, emb_dim:] * rel_1[:, :emb_dim]
            h_d1_for = (1/cov_anchor) * mu_d1_for

            mu_d2_for = lhs_2[:, emb_dim:] * rel_2[:, :emb_dim]
            h_d2_for = (1/cov_anchor) * mu_d2_for

            mu_d3_for = lhs_3[:, emb_dim:] * rel_3[:, :emb_dim]
            h_d3_for = (1/cov_anchor) * mu_d3_for

            h_u_for = h_u_for + h_d1_for + h_d2_for + h_d3_for
            J_u_for = (1/cov_anchor) + (1/cov_anchor) + (1/cov_anchor) + (1/cov_target)
            mu_u_for = h_u_for / J_u_for

            mu_d1_inv = lhs_1[:, :emb_dim] * rel_1[:, emb_dim:]
            h_d1_inv = (1/cov_anchor) * mu_d1_inv

            mu_d2_inv = lhs_2[:, :emb_dim] * rel_2[:, emb_dim:]
            h_d2_inv = (1/cov_anchor) * mu_d2_inv

            mu_d3_inv = lhs_3[:, :emb_dim] * rel_3[:, emb_dim:]
            h_d3_inv = (1/cov_anchor) * mu_d3_inv

            h_u_inv = h_u_inv + h_d1_inv + h_d2_inv + h_d3_inv
            J_u_inv = (1/cov_anchor) + (1/cov_anchor) + (1/cov_anchor) + (1/cov_target)
            mu_u_inv = h_u_inv / J_u_inv
        else:
            raise ValueError(f'Invalid number of intersections: {len(chains)}')

        
        #obj_guess = torch.normal(0, self.init_size, lhs_2.shape, device=lhs_2.device, requires_grad=True)
        #params = [obj_guess]

        #scores = self._optimize_variables(scoring_fn, params, optimizer, lr, max_steps)
        #return scores
        scores = []

        scores = torch.zeros( mu_u_for.shape[0], all_heads_embs.shape[0])
        if not disjunctive:
            for i in range(scores.shape[0]):
            ## get the dot product between row[i] of mu_u_for and each row of all_heads_embs
                scores [i] = torch.clip( 0.5*( all_heads_embs @ (mu_u_for[i]).T + all_tails_embs @ (mu_u_inv[i]).T), min=-20, max=20)
        
        elif disjunctive:
            scores1 = torch.zeros( mu_u_for1.shape[0], all_heads_embs.shape[0])
            scores2 = torch.zeros( mu_u_for2.shape[0], all_heads_embs.shape[0])
            for i in range(scores.shape[0]):
            ## get the dot product between row[i] of mu_u_for and each row of all_heads_embs
                scores1 [i] = torch.clip( 0.5*( all_heads_embs @ (mu_u_for1[i]).T + all_tails_embs @ (mu_u_inv1[i]).T), min=-20, max=20)
                scores2 [i] = torch.clip( 0.5*( all_heads_embs @ (mu_u_for2[i]).T + all_tails_embs @ (mu_u_inv2[i]).T), min=-20, max=20)
            scores = scores1 + scores2
        
        return scores

    def optimize_intersections(self, chains: List, regularizer: Regularizer,
                               max_steps: int = 20, lr: float = 0.1,
                               optimizer:str = 'adam', norm_type: str = 'min',
                               disjunctive=False):
        def scoring_fn(score_all=False):
            score_1, factors = self.score_emb(lhs_1, rel_1, obj_guess)
            guess_regularizer = regularizer([factors[2]])
            score_2, _ = self.score_emb(lhs_2, rel_2, obj_guess)

            atoms = torch.sigmoid(torch.cat((score_1, score_2), dim=1))

            if len(chains) == 3:
                score_3, _ = self.score_emb(lhs_3, rel_3, obj_guess)
                atoms = torch.cat((atoms, torch.sigmoid(score_3)), dim=1)

            if disjunctive:
                t_norm = self.batch_t_conorm(atoms, norm_type)
            else:
                t_norm = self.batch_t_norm(atoms, norm_type)

            all_scores = None
            if score_all:
                score_1 = self.forward_emb(lhs_1, rel_1)
                score_2 = self.forward_emb(lhs_2, rel_2)
                atoms = torch.stack((score_1, score_2), dim=-1)

                if disjunctive:
                    atoms = torch.sigmoid(atoms)

                if len(chains) == 3:
                    score_3 = self.forward_emb(lhs_3, rel_3)
                    atoms = torch.cat((atoms, score_3.unsqueeze(-1)), dim=-1)

                if disjunctive:
                    all_scores = self.batch_t_conorm(atoms, norm_type)
                else:
                    all_scores = self.batch_t_norm(atoms, norm_type)

            return t_norm, guess_regularizer, all_scores

        if len(chains) == 2:
            raw_chain = self.__get_chains__(chains, graph_type=QuerDAG.TYPE2_2.value)
            lhs_1, rel_1, lhs_2, rel_2 = raw_chain
        elif len(chains) == 3:
            raw_chain = self.__get_chains__(chains, graph_type=QuerDAG.TYPE2_3.value)
            lhs_1, rel_1, lhs_2, rel_2, lhs_3, rel_3 = raw_chain
        else:
            raise ValueError(f'Invalid number of intersections: {len(chains)}')

        obj_guess = torch.normal(0, self.init_size, lhs_2.shape, device=lhs_2.device, requires_grad=True)
        params = [obj_guess]

        scores = self._optimize_variables(scoring_fn, params, optimizer, lr, max_steps)
        return scores

    def optimize_3_3_bpl(self, chains: List, regularizer: Regularizer,
                               max_steps: int = 20, lr: float = 0.1,
                               optimizer:str = 'adam', norm_type: str = 'min',
                               cov_anchor: float = 0.1,
                        cov_var: float = 0.1, cov_target: float = 0.1, possible_heads_emb: list = None, possible_tails_emb: list = None,
                        all_nodes_embs: torch.tensor = None):
        #lhs_1 is a tensor of size (query_size, (2*)emb_dim)
        lhs_1, rel_1, rel_2, lhs_2, rel_3 = self.__get_chains__(chains, graph_type=QuerDAG.TYPE3_3.value)
        # updating the variable node given the first anchor node
        emb_dim = chains[0][0].shape[1] // 2
        all_heads_embs = all_nodes_embs[:, :emb_dim]
        all_tails_embs = all_nodes_embs[:, emb_dim:]
        mu_m_for = possible_tails_emb[0][:, :emb_dim]
        h_m_for = (1/cov_var) * mu_m_for
        mu_m_inv = possible_tails_emb[0][:, emb_dim:]
        h_m_inv = (1/cov_var) * mu_m_inv
        mu_d1_for = lhs_1[:, emb_dim:] * rel_1[:, :emb_dim]
        h_d1_for = (1/cov_anchor) * mu_d1_for
        h_m_for = h_m_for + h_d1_for
        J_m_for = (1/cov_anchor) + (1/cov_var)
        mu_d_inv = lhs_1[:, :emb_dim] * rel_1[:, emb_dim:]
        h_d_inv = (1/cov_anchor) * mu_d_inv
        h_m_inv = h_m_inv + h_d_inv
        J_m_inv = (1/cov_anchor) + (1/cov_var)

        # updating the target node given the variable node requires marginalization as in 2p, but not for the 
        # second anchor
        mu_u_for = possible_tails_emb[1][:, :emb_dim]
        h_u_for = (1/cov_target) * mu_u_for
        mu_u_inv = possible_tails_emb[1][:, emb_dim:]
        h_u_inv = (1/cov_target) * mu_u_inv

        mu_d2_for = lhs_2[:, emb_dim:] * rel_3[:, :emb_dim]
        h_d2_for = (1/cov_anchor) * mu_d2_for
        mu_d2_inv = lhs_2[:, :emb_dim] * rel_3[:, emb_dim:]
        h_d2_inv = (1/cov_anchor) * mu_d2_inv

        h_u_for = h_u_for + h_d2_for - rel_2[:, :emb_dim] * (1 / J_m_for) * h_m_for
        J_u_for = (1/cov_target) + (1/cov_anchor) - rel_2[:, :emb_dim] * (1 / J_m_for) * rel_2[:, :emb_dim]
        h_u_inv = h_u_inv + h_d2_inv - rel_2[:, emb_dim:] * (1 / J_m_inv) * h_m_inv
        J_u_inv = (1/cov_target) + (1/cov_anchor) - rel_2[:, emb_dim:] * (1 / J_m_inv) * rel_2[:, emb_dim:]
        mu_u_for = h_u_for / J_u_for
        mu_u_inv = h_u_inv / J_u_inv

        scores = []
        scores = torch.zeros( mu_u_for.shape[0], all_heads_embs.shape[0])
        for i in range(scores.shape[0]):
            ## get the dot product between row[i] of mu_u_for and each row of all_heads_embs
            scores [i] = torch.clip( 0.5*( all_heads_embs @ (mu_u_for[i]).T + all_tails_embs @ (mu_u_inv[i]).T), min=-20, max=20)

        return scores



    def optimize_3_3(self, chains: List, regularizer: Regularizer,
                     max_steps: int = 20, lr: float = 0.1,
                     optimizer:str = 'adam', norm_type: str = 'min'):
        def scoring_fn(score_all=False):
            score_1, factors_1 = self.score_emb(lhs_1, rel_1, obj_guess_1)
            score_2, _ = self.score_emb(obj_guess_1, rel_2, obj_guess_2)
            score_3, factors_2 = self.score_emb(lhs_2, rel_3, obj_guess_2)
            factors = [factors_1[2], factors_2[2]]

            atoms = torch.sigmoid(
                torch.cat((score_1, score_2, score_3), dim=1))

            guess_regularizer = regularizer(factors)

            t_norm = self.batch_t_norm(atoms, norm_type)

            all_scores = None
            if score_all:
                score_2 = self.forward_emb(obj_guess_1, rel_2)
                score_3 = self.forward_emb(lhs_2, rel_3)
                atoms = torch.sigmoid(torch.stack((score_1.expand_as(score_2), score_2, score_3), dim=-1))

                t_norm = self.batch_t_norm(atoms, norm_type)

                all_scores = t_norm

            return t_norm, guess_regularizer, all_scores

        lhs_1, rel_1, rel_2, lhs_2, rel_3 = self.__get_chains__(chains, graph_type=QuerDAG.TYPE3_3.value)

        obj_guess_1 = torch.normal(0, self.init_size, lhs_1.shape, device=lhs_1.device, requires_grad=True)
        obj_guess_2 = torch.normal(0, self.init_size, lhs_1.shape, device=lhs_1.device, requires_grad=True)
        params = [obj_guess_1, obj_guess_2]

        scores = self._optimize_variables(scoring_fn, params, optimizer, lr, max_steps)
        return scores

    def optimize_4_3_bpl(self, chains: List, regularizer: Regularizer,
                     max_steps: int = 20, lr: float = 0.1,
                     optimizer: str = 'adam', norm_type: str = 'min',
                     disjunctive=False, cov_anchor: float = 0.1,
                        cov_var: float = 0.1, cov_target: float = 0.1, possible_heads_emb: list = None, possible_tails_emb: list = None,
                        all_nodes_embs: torch.tensor = None):
        
        lhs_1, rel_1, lhs_2, rel_2, rel_3 = self.__get_chains__(chains, graph_type=QuerDAG.TYPE4_3.value)
        emb_dim = chains[0][0].shape[1] // 2
        all_heads_embs = all_nodes_embs[:, :emb_dim]
        all_tails_embs = all_nodes_embs[:, emb_dim:]
        mu_m_for = possible_tails_emb[0][:, :emb_dim]
        h_m_for = (1/cov_var) * mu_m_for
        mu_m_inv = possible_tails_emb[0][:, emb_dim:]
        h_m_inv = (1/cov_var) * mu_m_inv

        mu_d1_for = lhs_1[:, emb_dim:] * rel_1[:, :emb_dim]
        h_d1_for = (1/cov_anchor) * mu_d1_for
        mu_d2_for = lhs_2[:, emb_dim:] * rel_2[:, :emb_dim]
        h_d2_for = (1/cov_anchor) * mu_d2_for
        mu_d1_inv = lhs_1[:, :emb_dim] * rel_1[:, emb_dim:]
        h_d1_inv = (1/cov_anchor) * mu_d1_inv
        mu_d2_inv = lhs_2[:, :emb_dim] * rel_2[:, emb_dim:]
        h_d2_inv = (1/cov_anchor) * mu_d2_inv

        mu_u_for = possible_tails_emb[2][:, :emb_dim]
        h_u_for = (1/cov_target) * mu_u_for
        mu_u_inv = possible_tails_emb[2][:, emb_dim:]
        h_u_inv = (1/cov_target) * mu_u_inv

        scores = torch.zeros( mu_u_for.shape[0], all_heads_embs.shape[0])

        if not disjunctive:
            h_m_for = h_m_for + h_d1_for + h_d2_for
            J_m_for = (1/cov_var) + (1/cov_anchor) + (1/cov_anchor)
            h_m_inv = h_m_inv + h_d1_inv + h_d2_inv
            J_m_inv = (1/cov_var) + (1/cov_anchor) + (1/cov_anchor)
            h_u_for = h_u_for - rel_3[:, :emb_dim] * (1 / J_m_for) * h_m_for
            J_u_for = (1/cov_target) - rel_3[:, :emb_dim] * (1 / J_m_for) * rel_3[:, :emb_dim]
            h_u_inv = h_u_inv - rel_3[:, emb_dim:] * (1 / J_m_inv) * h_m_inv
            J_u_inv = (1/cov_target) - rel_3[:, emb_dim:] * (1 / J_m_inv) * rel_3[:, emb_dim:]

            mu_u_for = h_u_for / J_u_for
            mu_u_inv = h_u_inv / J_u_inv

            for i in range(scores.shape[0]):
                scores[i] =torch.clip( 0.5*( all_heads_embs @ (mu_u_for[i]).T + all_tails_embs @ (mu_u_inv[i]).T), min=-20, max=20)

        elif disjunctive:
            # finding first mu_u_for from the top chain
            h_m_for1 = h_m_for + h_d1_for
            J_m_for1 = (1/cov_var) + (1/cov_anchor)
            h_u_for1 = h_u_for - rel_3[:, :emb_dim] * (1 / J_m_for1) * h_m_for1
            J_u_for1 = (1/cov_target) - rel_3[:, :emb_dim] * (1 / J_m_for1) * rel_3[:, :emb_dim]
            mu_u_for1 = h_u_for1 / J_u_for1

            h_m_inv1 = h_m_inv + h_d1_inv
            J_m_inv1 = (1/cov_var) + (1/cov_anchor)
            h_u_inv1 = h_u_inv - rel_3[:, emb_dim:] * (1 / J_m_inv1) * h_m_inv1
            J_u_inv1 = (1/cov_target) - rel_3[:, emb_dim:] * (1 / J_m_inv1) * rel_3[:, emb_dim:]
            mu_u_inv1 = h_u_inv1 / J_u_inv1

            # finding second mu_u_for from the bottom chain
            h_m_for2 = h_m_for + h_d2_for
            J_m_for2 = (1/cov_var) + (1/cov_anchor)
            h_u_for2 = h_u_for - rel_3[:, :emb_dim] * (1 / J_m_for2) * h_m_for2
            J_u_for2 = (1/cov_target) - rel_3[:, :emb_dim] * (1 / J_m_for2) * rel_3[:, :emb_dim]
            mu_u_for2 = h_u_for2 / J_u_for2

            h_m_inv2 = h_m_inv + h_d2_inv
            J_m_inv2 = (1/cov_var) + (1/cov_anchor)
            h_u_inv2 = h_u_inv - rel_3[:, emb_dim:] * (1 / J_m_inv2) * h_m_inv2
            J_u_inv2 = (1/cov_target) - rel_3[:, emb_dim:] * (1 / J_m_inv2) * rel_3[:, emb_dim:]
            mu_u_inv2 = h_u_inv2 / J_u_inv2

            scores1 = torch.zeros( mu_u_for1.shape[0], all_heads_embs.shape[0])
            scores2 = torch.zeros( mu_u_for2.shape[0], all_heads_embs.shape[0])
            for i in range(scores.shape[0]):
                scores1[i] = torch.clip( 0.5*( all_heads_embs @ (mu_u_for1[i]).T + all_tails_embs @ (mu_u_inv1[i]).T), min=-20, max=20)
                scores2[i] = torch.clip( 0.5*( all_heads_embs @ (mu_u_for2[i]).T + all_tails_embs @ (mu_u_inv2[i]).T), min=-20, max=20)

            scores = scores1 + scores2

        return scores
        

    def optimize_4_3(self, chains: List, regularizer: Regularizer,
                     max_steps: int = 20, lr: float = 0.1,
                     optimizer: str = 'adam', norm_type: str = 'min',
                     disjunctive=False):
        def scoring_fn(score_all=False):
            score_1, factors_1 = self.score_emb(lhs_1, rel_1, obj_guess_1)
            score_2, _ = self.score_emb(lhs_2, rel_2, obj_guess_1)
            score_3, factors_2 = self.score_emb(obj_guess_1, rel_3,
                                                obj_guess_2)
            factors = [factors_1[2], factors_2[2]]
            guess_regularizer = regularizer(factors)

            if not disjunctive:
                atoms = torch.sigmoid(torch.cat((score_1, score_2, score_3), dim=1))
                t_norm = self.batch_t_norm(atoms, norm_type)
            else:
                disj_atoms = torch.sigmoid(torch.cat((score_1, score_2), dim=1))
                t_conorm = self.batch_t_conorm(disj_atoms, norm_type).unsqueeze(1)

                conj_atoms = torch.cat((t_conorm, torch.sigmoid(score_3)), dim=1)
                t_norm = self.batch_t_norm(conj_atoms, norm_type)

            all_scores = None
            if score_all:
                score_3 = self.forward_emb(obj_guess_1, rel_3)
                if not disjunctive:
                    atoms = torch.sigmoid(torch.stack((score_1.expand_as(score_3), score_2.expand_as(score_3), score_3), dim=-1))
                else:
                    atoms = torch.stack((t_conorm.expand_as(score_3), torch.sigmoid(score_3)), dim=-1)

                all_scores = self.batch_t_norm(atoms, norm_type)

            return t_norm, guess_regularizer, all_scores

        lhs_1, rel_1, lhs_2, rel_2, rel_3 = self.__get_chains__(chains, graph_type=QuerDAG.TYPE4_3.value)

        obj_guess_1 = torch.normal(0, self.init_size, lhs_1.shape, device=lhs_1.device, requires_grad=True)
        obj_guess_2 = torch.normal(0, self.init_size, lhs_1.shape, device=lhs_1.device, requires_grad=True)
        params = [obj_guess_1, obj_guess_2]

        scores = self._optimize_variables(scoring_fn, params, optimizer, lr, max_steps)
        return scores

    def get_best_candidates(self,
            rel: Tensor,
            arg1: Optional[Tensor],
            arg2: Optional[Tensor],
            candidates: int = 5,
            last_step = False, env: DynKBCSingleton = None) -> Tuple[Tensor, Tensor]:

        z_scores, z_emb, z_indices = None, None , None

        assert (arg1 is None) ^ (arg2 is None)

        batch_size, embedding_size = rel.shape[0], rel.shape[1]

        # [B, N]
        # scores_sp = (s, p, ?)
        # scores_sp, scores_po = self.candidates_score(rel, arg1, arg2)
        # scores = scores_sp if arg2 is None else scores_po

        scores = self.forward_emb(arg1, rel)

        if not last_step:
            # [B, K], [B, K]
            k = min(candidates, scores.shape[1])
            z_scores, z_indices = torch.topk(scores, k=k, dim=1)
            # [B, K, E]
            z_emb = self.entity_embeddings(z_indices)
            assert z_emb.shape[0] == batch_size
            assert z_emb.shape[2] == embedding_size

            if env is not None:
                logger = logging.getLogger('explain')
                logger.info(f'{"Rank":<6} {"X":<30} {"Score":<8}')
                for i in range(z_indices.shape[1]):
                    ent_id = z_indices[0, i].item()
                    ent_score = torch.sigmoid(z_scores[0, i]).item()
                    logger.info(f'{i:<6} {env.fb2name[env.ent_id2fb[ent_id]]:<30} {ent_score:<8.4f}')
        else:
            z_scores = scores

            z_indices = torch.arange(z_scores.shape[1]).view(1,-1).repeat(z_scores.shape[0],1).to(Device)
            z_emb = self.entity_embeddings(z_indices)

        return z_scores, z_emb

    def t_norm(self, tens_1: Tensor, tens_2: Tensor, t_norm: str = 'min') -> Tensor:
        if 'min' in t_norm:
            return torch.min(tens_1, tens_2)
        elif 'prod' in t_norm:
            return tens_1 * tens_2

    def t_conorm(self, tens_1: Tensor, tens_2: Tensor, t_conorm: str = 'max') -> Tensor:
        if 'min' in t_conorm:
            return torch.max(tens_1, tens_2)
        elif 'prod' in t_conorm:
            return (tens_1+tens_2) - (tens_1 * tens_2)

    def min_max_rescale(self, x):
        return (x-torch.min(x))/(torch.max(x) - torch.min(x))

    def query_answering_BF(self, env: DynKBCSingleton, candidates: int = 5, t_norm: str = 'min', batch_size = 1, scores_normalize = 0, explain=False):

        res = None
        # for disjunction, we need to use the t-conorm
        if 'disj' in env.graph_type:
            objective = self.t_conorm
        else:
            objective = self.t_norm

        chains, chain_instructions = env.chains, env.chain_instructions
        # chain_instructions = ['hop_0_1']
        # chains = [part1, part2]
        # part1 = [lhs_1, rels_1, rhs_1]
        # len(lhs_2) = 8000

        nb_queries, embedding_size = chains[0][0].shape[0], chains[0][0].shape[1]

        scores = None

        # data_loader = DataLoader(dataset=chains, batch_size=16, shuffle=False)

        batches = make_batches(nb_queries, batch_size)
        # batches = [(0,1), (1,2), (2,3), ...]

        explain = env.graph_type == '1_2' and explain
        if explain:
            logger = logging.getLogger('explain')
            logger.setLevel(logging.INFO)
            fh = logging.FileHandler('explain.log', mode='w')
            fh.setLevel(logging.INFO)
            logger.addHandler(fh)

        for i, batch in enumerate(tqdm.tqdm(batches)):
            if explain:
                query = env.keys_complete[i]
                anchor, rel1, x1, x2, rel2, x3 = query.split('_')
                anchor = env.fb2name[env.ent_id2fb[int(anchor)]]
                rel1 = env.rel_id2fb[int(rel1)]
                rel2 = env.rel_id2fb[int(rel2)]
                logger.info('-' * 100)
                logger.info(
                    f'Query: ?Y:∃ X.({anchor}, {rel1}, X) and (X, {rel2}, Y)')

            nb_branches = 1
            nb_ent = 0
            batch_scores = None
            candidate_cache = {}

            batch_size = batch[1] - batch[0]
            # here, batch_size is 1
            #torch.cuda.empty_cache()
            dnf_flag = False
            if 'disj' in env.graph_type:
                dnf_flag = True

            for inst_ind, inst in enumerate(chain_instructions):
                with torch.no_grad():
                    # this if for the case of projection
                    if 'hop' in inst:

                        ind_1 = int(inst.split("_")[-2])
                        ind_2 = int(inst.split("_")[-1])

                        indices = [ind_1, ind_2]
                        # indices = [0, 1]
                        # each index is one hop

                        if objective == self.t_conorm and dnf_flag:
                            objective = self.t_norm

                        last_hop = False
                        for hop_num, ind in enumerate(indices):

                            # print("HOP")
                            # print(candidate_cache.keys())
                            last_step =  (inst_ind == len(chain_instructions)-1) and last_hop


                            lhs,rel,rhs = chains[ind]

                            # [a, p, X], [X, p, Y][Y, p, Z]

                            # takes one of the lhs, rel (their embeddings)
                            if lhs is not None:
                                lhs = lhs[batch[0]:batch[1]]

                            else:
                                # print("MTA BRAT")
                                batch_scores, lhs_3d = candidate_cache[f"lhs_{ind}"]
                                lhs = lhs_3d.view(-1, embedding_size)
                            rel = rel[batch[0]:batch[1]]
                            rel = rel.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                            rel = rel.view(-1, embedding_size)
                            if f"rhs_{ind}" not in candidate_cache:
                                # gets best candidates for the rhs of this hop and the scores
                                z_scores, rhs_3d = self.get_best_candidates(rel, lhs, None, candidates, last_step, env if explain else None)
                                # z_scores : tensor of shape [Num_queries * Candidates^K]
                                # rhs_3d : tensor of shape [Num_queries, Candidates^K, Embedding_size]

                                # [Num_queries * Candidates^K]
                                z_scores_1d = z_scores.view(-1)
                                if 'disj' in env.graph_type or scores_normalize:
                                    z_scores_1d = torch.sigmoid(z_scores_1d)

                                # B * S
                                nb_sources = rhs_3d.shape[0]*rhs_3d.shape[1]
                                nb_branches = nb_sources // batch_size
                                # if the batch_score is None, we initialize it with the candidates scores (since there's just one hop). otherwise, the t-norm is applied
                                if not last_step:
                                    batch_scores = z_scores_1d if batch_scores is None else objective(z_scores_1d, batch_scores.view(-1, 1).repeat(1, candidates).view(-1), t_norm)
                                else:
                                    nb_ent = rhs_3d.shape[1]
                                    batch_scores = z_scores_1d if batch_scores is None else objective(z_scores_1d, batch_scores.view(-1, 1).repeat(1, nb_ent).view(-1), t_norm)
                                # candidate_cache stores the scores and the candidate embeddings for each rhs
                                candidate_cache[f"rhs_{ind}"] = (batch_scores, rhs_3d)
                                if not last_hop:
                                    #candidate_cache of the rhs of this hop is the lhs of the next hop
                                    candidate_cache[f"lhs_{indices[hop_num+1]}"] = (batch_scores, rhs_3d)

                            else:
                                # if we already have the rhs of this hop, we are in the last hop (so no more lhs)
                                batch_scores, rhs_3d = candidate_cache[f"rhs_{ind}"]
                                candidate_cache[f"lhs_{ind+1}"] = (batch_scores, rhs_3d)
                                last_hop =  True
                                del lhs, rel
                                # #torch.cuda.empty_cache()
                                continue

                            last_hop =  True
                            del lhs, rel, rhs, rhs_3d, z_scores_1d, z_scores
                            # #torch.cuda.empty_cache()

                    elif 'inter' in inst:
                        ind_1 = int(inst.split("_")[-2])
                        ind_2 = int(inst.split("_")[-1])

                        indices = [ind_1, ind_2]

                        if objective == self.t_norm and dnf_flag:
                            objective = self.t_conorm

                        if len(inst.split("_")) > 3:
                            ind_1 = int(inst.split("_")[-3])
                            ind_2 = int(inst.split("_")[-2])
                            ind_3 = int(inst.split("_")[-1])

                            indices = [ind_1, ind_2, ind_3]

                        for intersection_num, ind in enumerate(indices):
                            # print("intersection")
                            # print(candidate_cache.keys())

                            last_step =  (inst_ind == len(chain_instructions)-1) #and ind == indices[0]

                            lhs,rel,rhs = chains[ind]

                            if lhs is not None:
                                lhs = lhs[batch[0]:batch[1]]
                                lhs = lhs.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                                lhs = lhs.view(-1, embedding_size)

                            else:
                                batch_scores, lhs_3d = candidate_cache[f"lhs_{ind}"]
                                lhs = lhs_3d.view(-1, embedding_size)
                                nb_sources = lhs_3d.shape[0]*lhs_3d.shape[1]
                                nb_branches = nb_sources // batch_size

                            rel = rel[batch[0]:batch[1]]
                            rel = rel.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                            rel = rel.view(-1, embedding_size)

                            if intersection_num > 0 and 'disj' in env.graph_type:
                                batch_scores, rhs_3d = candidate_cache[f"rhs_{ind}"]
                                rhs = rhs_3d.view(-1, embedding_size)
                                z_scores = self.score_fixed(rel, lhs, rhs, candidates)

                                z_scores_1d = z_scores.view(-1)
                                if 'disj' in env.graph_type or scores_normalize:
                                    z_scores_1d = torch.sigmoid(z_scores_1d)

                                batch_scores = z_scores_1d if batch_scores is None else objective(z_scores_1d, batch_scores, t_norm)

                                continue

                            if f"rhs_{ind}" not in candidate_cache or last_step:
                                z_scores, rhs_3d = self.get_best_candidates(rel, lhs, None, candidates, last_step, env if explain else None)

                                # [B * Candidates^K] or [B, S-1, N]
                                z_scores_1d = z_scores.view(-1)
                                # print(z_scores_1d)
                                if 'disj' in env.graph_type or scores_normalize:
                                    z_scores_1d = torch.sigmoid(z_scores_1d)

                                if not last_step:
                                    nb_sources = rhs_3d.shape[0]*rhs_3d.shape[1]
                                    nb_branches = nb_sources // batch_size

                                if not last_step:
                                    batch_scores = z_scores_1d if batch_scores is None else objective(z_scores_1d, batch_scores.view(-1, 1).repeat(1, candidates).view(-1), t_norm)
                                else:
                                    if ind == indices[0]:
                                        nb_ent = rhs_3d.shape[1]
                                    else:
                                        nb_ent = 1

                                    batch_scores = z_scores_1d if batch_scores is None else objective(z_scores_1d, batch_scores.view(-1, 1).repeat(1, nb_ent).view(-1), t_norm)
                                    nb_ent = rhs_3d.shape[1]

                                candidate_cache[f"rhs_{ind}"] = (batch_scores, rhs_3d)

                                if ind == indices[0] and 'disj' in env.graph_type:
                                    count = len(indices)-1
                                    iterator = 1
                                    while count > 0:
                                        candidate_cache[f"rhs_{indices[intersection_num+iterator]}"] = (batch_scores, rhs_3d)
                                        iterator += 1
                                        count -= 1

                                if ind == indices[-1]:
                                    candidate_cache[f"lhs_{ind+1}"] = (batch_scores, rhs_3d)
                            else:
                                batch_scores, rhs_3d = candidate_cache[f"rhs_{ind}"]
                                candidate_cache[f"rhs_{ind+1}"] = (batch_scores, rhs_3d)

                                last_hop = True
                                del lhs, rel
                                continue

                            del lhs, rel, rhs, rhs_3d, z_scores_1d, z_scores

            if batch_scores is not None:
                # [B * entites * S ]
                # S ==  K**(V-1)

                scores_2d = batch_scores.view(batch_size,-1, nb_ent )
                #print(scores_2d.shape)
                # [1,candidates, nb_ent]
                # res is the max score for each entity among the candidates
                res, _ = torch.max(scores_2d, dim=1)
                


                if explain:
                    final_scores = scores_2d.squeeze()
                    for j in range(final_scores.shape[0]):
                        logger.info(f'X = {j}')

                        z_scores, z_indices = torch.topk(final_scores, k=candidates, dim=1)

                        logger.info(f'\t{"Rank":<6} {"Y":<30} {"Final score":<12}')
                        for k in range(z_indices.shape[1]):
                            ent_id = z_indices[j, k].item()
                            ent_score = z_scores[j, k].item()
                            logger.info(f'\t{k:<6} {env.fb2name[env.ent_id2fb[ent_id]]:<30} {ent_score:<8.4f}')

                    test_answers = set(env.target_ids_complete[query])

                    res_top_val, res_top_idx = torch.topk(res.squeeze(), k=candidates)
                    logger.info(f'Top {candidates} final answers')
                    logger.info(f'{"Rank":<6} {"Y":<30} {"Final score":<12}')
                    for j in range(res_top_val.shape[0]):
                        ent_id = res_top_idx[j].item()
                        ent_score = res_top_val[j].item()
                        correct = '✓' if ent_id in test_answers else '✗'
                        logger.info(f'[{correct}] {j:<6} {env.fb2name[env.ent_id2fb[ent_id]]:<30} {ent_score:<8.4f}')

                    logger.info(f'Ground truth answers')
                    for ans in test_answers:
                        logger.info(f'- {env.fb2name[env.ent_id2fb[ans]]}')

                scores = res if scores is None else torch.cat([scores,res])

                del batch_scores, scores_2d, res,candidate_cache

            else:
                assert False, "Batch Scores are empty: an error went uncaught."

            res = scores

        if explain:
            fh.close()

        # res has the score of each entity for each query
        return res
class SimplE(KBCModel):
    def __init__(
        self, sizes: Tuple[int, int, int], rank: int,
        init_size: float = 1e-3
        ):
        super(SimplE, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

        self.init_size = init_size

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        for_prod = torch.sum(lhs[0] * rel[0] * rhs[1], 1, keepdim=True)
        inv_prod = torch.sum(lhs[1] * rel[1] * rhs[0], 1, keepdim=True)
        return torch.clamp((for_prod + inv_prod)/2, min=-20, max=20)

    def entity_embeddings(self, indices: Tensor):
        return self.embeddings[0](indices)

    def score_fixed(self, rel: Tensor, arg1: Tensor, arg2: Tensor,
                    *args, **kwargs) -> Tensor:
        rel_f, rel_inv = rel[:, :self.rank], rel[:, self.rank:]
        arg1_f, arg1_inv = arg1[:, :self.rank], arg1[:, self.rank:]
        arg2_f, arg2_inv = arg2[:, :self.rank], arg2[:, self.rank:]

        score_f = torch.sum(arg1_f * rel_f * arg2_inv, 1, keepdim=True)
        score_inv = torch.sum(arg1_inv * rel_inv * arg2_f, 1, keepdim=True)
        res = torch.clamp((score_f + score_inv)/2, min=-20, max=20)
        del rel_f, rel_inv, arg1_f, arg1_inv, arg2_f, arg2_inv, score_f, score_inv
        return res


# TODO: Check if this is correct
    def candidates_score(self,
                    rel: Tensor,
                arg1: Optional[Tensor],
                arg2: Optional[Tensor],
                *args, **kwargs) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        emb = self.embeddings[0].weight
        rel_f, rel_inv = rel[:, :self.rank], rel[:, self.rank:]
        emb_f, emb_inv = emb[:, :self.rank], emb[:, self.rank:]

        score_sp = score_po = None
        # calculates score of <arg1, rel, ent> triples for all entities
        if arg1 is not None:
            arg1_f, arg1_inv = arg1[:, :self.rank], arg1[:, self.rank:]
            score_f_sp = (rel_f * arg1_f) @ emb_inv.t()
            score_inv_sp = (rel_inv * arg1_inv) @ emb_f.t()
            score_sp = torch.clamp((score_f_sp + score_inv_sp)/2, min=-20, max=20)
        # calculates scores of <ent, rel, arg2> triples for all entities
        if arg2 is not None:
            arg2_f, arg2_inv = arg2[:, :self.rank], arg2[:, self.rank:]
            score_f_po = (rel_f * arg2_f) @ emb_inv.t()
            score_inv_po = (rel_inv * arg2_inv) @ emb_f.t()
            score_po = torch.clamp((score_f_po + score_inv_po)/2, min=-20, max=20)

        return score_sp, score_po
# returns the score of the given triples and regularization losses for head, rel, and tail
    def score_emb(self, lhs_emb, rel_emb, rhs_emb):
        lhs = lhs_emb[:, :self.rank], lhs_emb[:, self.rank:]
        rel = rel_emb[:, :self.rank], rel_emb[:, self.rank:]
        rhs = rhs_emb[:, :self.rank], rhs_emb[:, self.rank:]
        for_prod = torch.sum(lhs[0] * rel[0] * rhs[1], 1, keepdim=True)
        inv_prod = torch.sum(lhs[1] * rel[1] * rhs[0], 1, keepdim=True)
        score = torch.clamp((for_prod + inv_prod)/2, min=-20, max=20)

        return score, (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )
        
    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        to_score = self.embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
        # to_score is all entities in tail
        for_prod = (lhs[0] * rel[0]) @ to_score[1].transpose(0, 1)
        inv_prod = (lhs[1] * rel[1]) @ to_score[0].transpose(0, 1)
        return torch.clamp((for_prod + inv_prod)/2, min=-20, max=20), (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )

    def forward_emb(self, lhs, rel):
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        to_score = self.embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
        # to_score is all entities in tail
        for_prod = (lhs[0] * rel[0]) @ to_score[1].transpose(0, 1)
        inv_prod = (lhs[1] * rel[1]) @ to_score[0].transpose(0, 1)
        return torch.clamp((for_prod + inv_prod)/2, min=-20, max=20)


    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries_separated(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])

        return (lhs, rel)

    def get_full_embeddigns(self, queries: torch.Tensor):

        if torch.sum(queries[:, 0]).item() > 0:
            lhs = self.embeddings[0](queries[:, 0])
        else:
            lhs = None

        if torch.sum(queries[:, 1]).item() > 0:

            rel = self.embeddings[1](queries[:, 1])
        else:
            rel = None

        if torch.sum(queries[:, 2]).item() > 0:
            rhs = self.embeddings[0](queries[:, 2])
        else:
            rhs = None

        return (lhs,rel,rhs)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        return torch.cat([
            0.5 * lhs[1] * rel[1],
            0.5 * lhs[0] * rel[0]
        ], 1)

    def model_type(self):
        return "SimplE"

class CP(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(CP, self).__init__()

        self.sizes = sizes
        self.rank = rank

        self.lhs = nn.Embedding(sizes[0], rank, sparse=True)
        self.rel = nn.Embedding(sizes[1], rank, sparse=True)
        self.rhs = nn.Embedding(sizes[2], rank, sparse=True)

        self.lhs.weight.data *= init_size
        self.rel.weight.data *= init_size
        self.rhs.weight.data *= init_size

    def entity_embeddings(self, indices: Tensor):
        return self.rhs(indices)

    def score(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])

        return torch.sum(lhs * rel * rhs, 1, keepdim=True)

    def score_emb(self, lhs, rel, rhs):
        return torch.mean(torch.sum(lhs * rel * rhs, 1, keepdim=True))

    def forward(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])
        return (lhs * rel) @ self.rhs.weight.t(), (lhs, rel, rhs)

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.rhs.weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries_separated(self, x: torch.Tensor):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])

        return (lhs,rel)

    def get_full_embeddigns(self, queries: torch.Tensor):
        if torch.sum(queries[:, 0]).item() > 0:
            lhs = self.lhs(queries[:, 0])
        else:
            lhs = None

        if torch.sum(queries[:, 1]).item() > 0:
            rel = self.rel(queries[:, 1])
        else:
            rel = None

        if torch.sum(queries[:, 2]).item() > 0:
            rhs = self.rhs(queries[:, 2])
        else:
            rhs = None

        return (lhs,rel,rhs)

    def get_queries(self, queries: torch.Tensor):
        return self.lhs(queries[:, 0]).data * self.rel(queries[:, 1]).data

    def model_type(self):
        return 'CP'


class ComplEx(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(ComplEx, self).__init__()

        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

        self.init_size = init_size

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        return torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1, keepdim=True
        )

    def entity_embeddings(self, indices: Tensor):
        return self.embeddings[0](indices)

    def score_fixed(self, rel: Tensor, arg1: Tensor, arg2: Tensor,
                    *args, **kwargs) -> Tensor:
        # [B, E]
        rel_real, rel_img = rel[:, :self.rank], rel[:, self.rank:]
        arg1_real, arg1_img = arg1[:, :self.rank], arg1[:, self.rank:]
        arg2_real, arg2_img = arg2[:, :self.rank], arg2[:, self.rank:]

        # [B] Tensor
        score1 = torch.sum(rel_real * arg1_real * arg2_real, 1)
        score2 = torch.sum(rel_real * arg1_img * arg2_img, 1)
        score3 = torch.sum(rel_img * arg1_real * arg2_img, 1)
        score4 = torch.sum(rel_img * arg1_img * arg2_real, 1)

        res = score1 + score2 + score3 - score4

        del score1,score2, score3, score4, rel_real, rel_img, arg1_real, arg1_img, arg2_real, arg2_img

        return res

    def candidates_score(self,
                rel: Tensor,
                arg1: Optional[Tensor],
                arg2: Optional[Tensor],
                *args, **kwargs) -> Tuple[Optional[Tensor], Optional[Tensor]]:

        emb = self.embeddings[0].weight

        rel_real, rel_img = rel[:, :self.rank], rel[:, self.rank:]
        emb_real, emb_img = emb[:, :self.rank], emb[:, self.rank:]

        # [B] Tensor

        score_sp = score_po = None

        if arg1 is not None:
            arg1_real, arg1_img = arg1[:, :self.rank], arg1[:, self.rank:]

            score1_sp = (rel_real * arg1_real) @ emb_real.t()
            score2_sp = (rel_real * arg1_img) @ emb_img.t()
            score3_sp = (rel_img * arg1_real) @ emb_img.t()
            score4_sp = (rel_img * arg1_img) @ emb_real.t()

            score_sp = score1_sp + score2_sp + score3_sp - score4_sp

        if arg2 is not None:
            arg2_real, arg2_img = arg2[:, :self.rank], arg2[:, self.rank:]

            score1_po = (rel_real * arg2_real) @ emb_real.t()
            score2_po = (rel_real * arg2_img) @ emb_img.t()
            score3_po = (rel_img * arg2_img) @ emb_real.t()
            score4_po = (rel_img * arg2_real) @ emb_img.t()

            score_po = score1_po + score2_po + score3_po - score4_po

        return score_sp, score_po

    def score_emb(self, lhs_emb, rel_emb, rhs_emb):
        lhs = lhs_emb[:, :self.rank], lhs_emb[:, self.rank:]
        rel = rel_emb[:, :self.rank], rel_emb[:, self.rank:]
        rhs = rhs_emb[:, :self.rank], rhs_emb[:, self.rank:]

        return torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1, keepdim=True), (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        to_score = self.embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
        return (
            (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1)
        ), (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )

    def forward_emb(self, lhs, rel):
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        to_score = self.embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
        return ((lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
                (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1))

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries_separated(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])

        return (lhs, rel)

    def get_full_embeddigns(self, queries: torch.Tensor):

        if torch.sum(queries[:, 0]).item() > 0:
            lhs = self.embeddings[0](queries[:, 0])
        else:
            lhs = None

        if torch.sum(queries[:, 1]).item() > 0:

            rel = self.embeddings[1](queries[:, 1])
        else:
            rel = None

        if torch.sum(queries[:, 2]).item() > 0:
            rhs = self.embeddings[0](queries[:, 2])
        else:
            rhs = None

        return (lhs,rel,rhs)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        return torch.cat([
            lhs[0] * rel[0] - lhs[1] * rel[1],
            lhs[0] * rel[1] + lhs[1] * rel[0]
        ], 1)

    def model_type(self):
        return "ComplEx"


class DistMult(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(DistMult, self).__init__()

        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([nn.Embedding(s, rank, sparse=True) for s in sizes[:2]])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

        self.init_size = init_size

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        return torch.sum(lhs * rel * rhs, 1, keepdim=True)

    def entity_embeddings(self, indices: Tensor):
        return self.embeddings[0](indices)

    def score_fixed(self, rel: Tensor, arg1: Tensor, arg2: Tensor,
                    *args, **kwargs) -> Tensor:
        return torch.sum(rel * arg1 * arg2, 1)

    def candidates_score(self,
                rel: Tensor,
                arg1: Optional[Tensor],
                arg2: Optional[Tensor],
                *args, **kwargs) -> Tuple[Optional[Tensor], Optional[Tensor]]:

        emb = self.embeddings[0].weight

        # [B] Tensor

        score_sp = score_po = None

        if arg1 is not None:
            score_sp = (rel * arg1) @ emb.t()

        if arg2 is not None:
            score_po = (rel * arg2) @ emb.t()

        return score_sp, score_po

    def score_emb(self, lhs_emb, rel_emb, rhs_emb):
        return (torch.sum(lhs_emb * rel_emb * rhs_emb[0], 1, keepdim=True),
                (lhs_emb, rel_emb, rhs_emb))

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        to_score = self.embeddings[0].weight
        return ((lhs * rel) @ to_score.transpose(0, 1),
                (lhs, rel, rhs))

    def forward_emb(self, lhs, rel):
        to_score = self.embeddings[0].weight
        return (lhs * rel) @ to_score.transpose(0, 1)

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries_separated(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])

        return (lhs, rel)

    def get_full_embeddigns(self, queries: torch.Tensor):

        if torch.sum(queries[:, 0]).item() > 0:
            lhs = self.embeddings[0](queries[:, 0])
        else:
            lhs = None

        if torch.sum(queries[:, 1]).item() > 0:

            rel = self.embeddings[1](queries[:, 1])
        else:
            rel = None

        if torch.sum(queries[:, 2]).item() > 0:
            rhs = self.embeddings[0](queries[:, 2])
        else:
            rhs = None

        return (lhs,rel,rhs)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])

        return lhs * rel

    def model_type(self):
        return "DistMult"
