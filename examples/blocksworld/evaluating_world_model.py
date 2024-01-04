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
max_steps = 6


# print(raw_example[1])
# def calc_acc(full_dataset, data_path, idx):
    # result = {}

    # action0 = plan.strip().split('\n')[0]
    # # print('Action:', action0)
    # world_model.update_example(example=example)
    # init_state = world_model.init_state()
    # # print('Initial State:', init_state.blocks_state)
    


    # # Get Predicted State
    # predicted_state, _  = world_model.step(init_state, action0)
    # predicted_state = predicted_state.blocks_state
    # # print('Predicted State:', predicted_state)

    # # lm_plan_file = "tmp_plan.txt"
    # # bw_utils.text_to_plan_blocksworld(plan, example["instance_file"], config_file, domain_file, lm_plan_file)

    # # open(lm_plan_file, mode='w', encoding='utf').write(action0 + '\n[PLAN END]')
    # # status, _ = bw_utils.validate_plan(domain_file, example["instance_file"], lm_plan_file)
    # # print(example['instance_file'])
    # # states = bw_utils.get_intermediate_states(domain_path=domain_file, instance=raw_example, config_data=bw_utils.read_config(config_file))
    # states = bw_utils.custom_get_intermediate_states(
    #     domain_path=domain_file,
    #     problem_path=problem_path,
    #     plan_code=plan_code,
    #     config_data=bw_utils.read_config(config_file),
    # )
    # gold_state = states[0]

    # # print()
    # # print('Gold State:', gold_state)

    # # print(type(gold_state), type(predicted_state))
    # # print('Accuracy:', predicted_state == gold_state)
    # result.update({'initial_state': init_state.blocks_state, 'action': action0, 'predicted_state': predicted_state, 'gold_state': gold_state, 'accuracy': predicted_state == gold_state})
    # return result


def predict_states(full_dataset, data_path, idx):
    result = {}
    # from reasoners.lm import GPTCompletionModel, 
    from reasoners.lm.openai_model_multikey import MultiKeyGPTCompletionModel
    # key = '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'
    # os.environ["OPENAI_API_KEY"] = key
    gpt_model = MultiKeyGPTCompletionModel(
        model='text-davinci-003',
        )
    world_model = BlocksWorldModel(base_model=gpt_model, prompt=prompt, batch_size=batch_size, max_steps=max_steps)
    example = full_dataset[idx]
    raw_example = json.load(open(data_path))[idx]
    result.update(example)
    problem_path, plan_code, _ = raw_example
    result.update({'plan_code': plan_code})
    plan = example['plan']
    actions = plan.strip().split('\n')[:-1]
    world_model.update_example(example=example)
    init_state = world_model.init_state()
    cur_state = init_state
    predicted_states = []
    for action in actions:
        predicted_state, _  = world_model.step(cur_state, action)
        cur_state = predicted_state
        predicted_state = predicted_state.blocks_state
        predicted_states.append(predicted_state)
    return predicted_states


if __name__ == "__main__":
    result_name = 'step_4_result.json'
    data_path: str = f'{ROOT_DIR}/data/full_data/step_4.json'
    data = json.load(open(f'{ROOT_DIR}/new_data/full_data_processed/step_4.json'))
    full_dataset = bw_utils.load_blocksworld(config_file, domain_file, data_list=data['raw_data'])  # [{"goal": str, "init": str}]
    # full_dataset = full_dataset[:5]
    print('Number of Examples:', len(full_dataset))
    results = []
    for idx in tqdm(range(len(full_dataset))):
        predicted_states = predict_states(full_dataset, data_path, idx)
        results.append(predicted_states)
        gold_states = data['gold_states'][idx]
        # print([a == b for a, b in zip(gold_states, predicted_states)])
        # break
    data.update({'predicted_states': results})
    json.dump(obj=data, fp=open(f'./results/{result_name}', 'w', encoding='utf'), indent=4)
    acc_list = [all([a == b for a, b in zip(ps, gs)]) for ps, gs in zip(data['gold_states'], data['predicted_states'])]
    print(acc_list)
    print('Average Acc:', sum(acc_list)/len(acc_list))
    # json.dump(obj=results, fp=open(f"./results/{result_name}", 'w', encoding='utf'), indent=4)