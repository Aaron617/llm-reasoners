import pickle
from typing import Type, Callable, Optional

import numpy as np
from tqdm import tqdm
from datetime import datetime

from reasoners import LanguageModel, Reasoner, SearchAlgorithm
from reasoners.benchmark import BWEvaluator
from reasoners.algorithm import MCTS

from world_model import BlocksWorldModel
from search_config import BWConfig
import os, json
import reasoners.benchmark.bw_utils as bw_utils

ROOT_DIR = os.path.join(os.path.dirname(__file__))


import reasoners.benchmark.bw_utils as bw_utils



prompt_path: str = f'{ROOT_DIR}/prompts/prompt.json'
config_file: str = f"{ROOT_DIR}/new_data/bw_config.yaml"
domain_file: str = f"{ROOT_DIR}/data/generated_domain.pddl"
os.environ['VAL'] = './LLMs-Planning/planner_tools/VAL/bin/MacOSExecutables'
with open(prompt_path) as f:
    prompt = json.load(f)
batch_size = 1
depth_limit = 2
# world_model = BlocksWorldModel(base_model=get_base_model(), prompt=prompt, batch_size=batch_size, max_steps=depth_limit)


# print(raw_example[1])
def get_states(full_dataset, data_path, idx):
    result = {}
    example = full_dataset[idx]
    raw_example = json.load(open(data_path))[idx]
    result.update(example)
    problem_path, plan_code, _ = raw_example
    result.update({'plan_code': plan_code})
    plan = example['plan']
    # action0 = plan.strip().split('\n')[0]
    
    states = bw_utils.custom_get_intermediate_states(
        domain_path=domain_file,
        problem_path=problem_path,
        plan_code=plan_code,
        config_data=bw_utils.read_config(config_file),
    )
    return states


if __name__ == "__main__":
    # result_name = 'step_4_result.json'
    for _ in [2, 4, 6, 8, 10, 12, 14, 16]:
        data_file = f'step_{_}.json'
        data_path: str = f'{ROOT_DIR}/data/full_data/{data_file}'
        out_data_dir = f"{ROOT_DIR}/new_data/full_data_processed"
        os.makedirs(out_data_dir, exist_ok=True)
        data_list = json.load(open(data_path))
        full_dataset = bw_utils.load_blocksworld(config_file, domain_file, data_list=data_list)  # [{"goal": str, "init": str}]
        # full_dataset = full_dataset[:5]
        print('Number of Examples:', len(full_dataset))
        # results = []
        state_list = []
        for idx in tqdm(range(len(full_dataset))):
            # results.append(calc_acc(full_dataset, data_path, idx))
            state_list.append(get_states(full_dataset, data_path, idx))
        json.dump(obj={'raw_data':data_list, 'gold_states':state_list}, fp=open(f'{out_data_dir}/{data_file}', 'w', encoding='utf'), indent=4)
    # acc_list = [_['accuracy'] for _ in results]
    # print('Average Acc:', sum(acc_list)/len(acc_list))
    # json.dump(obj=results, fp=open(f"./results/{result_name}", 'w', encoding='utf'), indent=4)çç