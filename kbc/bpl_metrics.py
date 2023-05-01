import torch
import numpy as np
from tqdm import tqdm
import logging
import sys


def explain_query(rel_id, ent_id, top_var_inds_list, top_target_inds, raw, query_id):
    raw_chain = raw[query_id].data['raw_chain']
    conv_raw_chain = 1 * raw_chain
    
    for ind in range(len(raw_chain)):
        part = raw_chain[ind]
        conv_raw_chain[ind][1] = rel_id[part[1]]
    # replacing the anchor node
    conv_raw_chain[0][0] = ent_id[raw_chain[0][0]]
    # replacing the variables of the first part 
    if len(conv_raw_chain) == 2:
        top_vars_0 = []
        for var in top_var_inds_list[0][query_id]:
            top_vars_0.append(ent_id[int(var)])
        conv_raw_chain[0][2] = top_vars_0
        conv_raw_chain[1][0] = conv_raw_chain[0][2]

        cov_gt_answers = []
        gt_answers = conv_raw_chain[-1][2]
        #cheking if the top targets are in the gt answers
        gt_array = np.array(gt_answers).astype(int)
        top_targets_array = np.array(top_target_inds[query_id]).astype(int)
        success = np.intersect1d(gt_array, top_targets_array).shape[0] > 0

        # converting ground truth answers to entity ids
        for ent in gt_answers:
            cov_gt_answers.append(ent_id[int(ent)])

        top_targets = []
        for  ent in (top_target_inds[query_id]):
            top_targets.append(ent_id[int(ent)])
        conv_raw_chain[-1][2] = top_targets

    if len(conv_raw_chain) == 3:
        top_vars_0 = []
        for var in top_var_inds_list[0][query_id]:
            top_vars_0.append(ent_id[int(var)])
        conv_raw_chain[0][2] = top_vars_0
        conv_raw_chain[1][0] = conv_raw_chain[0][2]

        top_vars_middle = []
        for var in top_var_inds_list[1][query_id]:
            top_vars_middle.append(ent_id[int(var)])
        conv_raw_chain[1][2] = top_vars_middle
        conv_raw_chain[2][0] = conv_raw_chain[1][2]

        top_targets = []
        cov_gt_answers = []
        gt_answers = conv_raw_chain[-1][2]
        #cheking if the top targets are in the gt answers
        gt_array = np.array(gt_answers).astype(int)
        top_targets_array = np.array(top_target_inds[query_id]).astype(int)
        success = np.intersect1d(gt_array, top_targets_array).shape[0] > 0

        # converting ground truth answers to entity ids
        for ent in gt_answers:
            cov_gt_answers.append(ent_id[int(ent)])
        
        # converting top target inds to entity ids
        for ent in (top_target_inds[query_id]):
            top_targets.append(ent_id[int(ent)])
        conv_raw_chain[-1][2] = top_targets

    
    #write the results to a text file

    if success:
        f = open("explain_success_1_{}.txt".format(len(conv_raw_chain)), "a")
    else:
        f = open("explain_fail_1_{}.txt".format(len(conv_raw_chain)), "a")
    f.write("query id:"+ str(query_id))
    f.write("\n")
    f.write("raw chain:"+"\t"+ str(conv_raw_chain))
    f.write("\n")
    f.write("ground truth answers:"+"\t"+ str(cov_gt_answers))
    f.write("\n")



def evaluation(scores, queries, test_ans, test_ans_hard, rel_id, ent_id, explain, top_var_inds_list, top_target_inds, raw):
    
    
    
    
    nentity = len(scores[0])
    step = 0
    logs = []

    

    for query_id, query in enumerate(tqdm(queries[:100])):
        if explain == 'yes':
            explain_query(rel_id, ent_id, top_var_inds_list, top_target_inds, raw, query_id)


        score = scores[query_id]
        score -= (torch.min(score) - 1)
        ans = test_ans[query]
        hard_ans = test_ans_hard[query]

        all_idx = set(range(nentity))

        false_ans = all_idx - set(ans)
        ans_list = list(ans)
        hard_ans_list = list(hard_ans)
        false_ans_list = list(false_ans)
        ans_idxs = np.array(hard_ans_list)
        vals = np.zeros((len(ans_idxs), nentity))

        vals[np.arange(len(ans_idxs)), ans_idxs] = 1
        axis2 = np.tile(false_ans_list, len(ans_idxs))

        # axis2 == [not_ans_1,...not_ans_k, not_ans_1, ....not_ans_k........]
        # Goes for len(hard_ans) times

        axis1 = np.repeat(range(len(ans_idxs)), len(false_ans))

        vals[axis1, axis2] = 1
        b = torch.tensor(vals, device=scores.device)
        filter_score = b * score
        # get number of elements in b that are not equal to 1
        argsort = torch.argsort(filter_score, dim=1, descending=True)
        ans_tensor = torch.tensor(hard_ans_list, device=scores.device, dtype=torch.long)
        # after the next line, only our found answers will become zero and its location will be the rank
        argsort = torch.transpose(torch.transpose(argsort, 0, 1) - ans_tensor, 0, 1)
        ranking = (argsort == 0).nonzero(as_tuple=False)

        ranking = ranking[:, 1]
        ranking = ranking + 1

        ans_vec = np.zeros(nentity)
        ans_vec[ans_list] = 1
        hits1m = torch.mean((ranking <= 1).to(torch.float)).item()
        hits3m = torch.mean((ranking <= 3).to(torch.float)).item()
        hits10m = torch.mean((ranking <= 10).to(torch.float)).item()
        mrm = torch.mean(ranking.to(torch.float)).item()
        mrrm = torch.mean(1./ranking.to(torch.float)).item()
        num_ans = len(hard_ans_list)

        hits1m_newd = hits1m
        hits3m_newd = hits3m
        hits10m_newd = hits10m
        mrm_newd = mrm
        mrrm_newd = mrrm

        logs.append({
            'MRRm_new': mrrm_newd,
            'MRm_new': mrm_newd,
            'HITS@1m_new': hits1m_newd,
            'HITS@3m_new': hits3m_newd,
            'HITS@10m_new': hits10m_newd,
            'num_answer': num_ans
        })

        if step % 100 == 0:
            logging.info('Evaluating the model... (%d/%d)' % (step, 1000))

        step += 1

    metrics = {}
    num_answer = sum([log['num_answer'] for log in logs])
    for metric in logs[0].keys():
        if metric == 'num_answer':
            continue
        if 'm' in metric:
            metrics[metric] = sum([log[metric] for log in logs])/len(logs)
        else:
            metrics[metric] = sum([log[metric] for log in logs])/num_answer

    return metrics