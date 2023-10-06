import io
import re
from typing import TypedDict, Optional
import numpy as np
from regex import P
from world_model import GSM8kState, GSM8kAction, GSM8kPrompt
from reasoners import SearchConfig, LanguageModel

class GSM8kUsefulPrompt(TypedDict):
    input: str
    question_prefix: str
    subquestion_prefix: str
    new_subquestion_prefix: str
    useful_prefix: str

class GSM8kConfig(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 useful_prompt: dict,
                 n_actions=4,
                 batch_size=1,
                 temperature=0.8,
                 reward_alpha=0.5,
                 reward_confidence_default=0.8,
                 depth_limit=5,
                 force_terminating_on_depth_limit=True,
                 force_overall_prompt_on_overall_question=True,
                 force_overall_question_on_overall_prompt=True) -> None:
        super().__init__()
        self.base_model = base_model
        self.example = ''
        self.prompt: GSM8kPrompt = prompt
        self.useful_prompt: GSM8kUsefulPrompt = useful_prompt
        self.batch_size = batch_size
        self.temperature = temperature
        self.n_actions = n_actions #n_actions = depth_limit - 1 is reasonable: actually there can be depth-1 actions be meaningful
        self.force_terminating_on_depth_limit = force_terminating_on_depth_limit
        self.depth_limit = depth_limit
        self.reward_alpha = reward_alpha
        self.reward_confidence_default = reward_confidence_default
        self.force_overall_prompt_on_overall_question = force_overall_prompt_on_overall_question
        self.force_overall_question_on_overall_prompt = force_overall_question_on_overall_prompt
        self.overall_question: Optional[str] = None

    def update_example(self, example: str) -> None:
        super().update_example(example)
        if self.force_overall_prompt_on_overall_question or self.force_overall_question_on_overall_prompt:
            # self.overall_question = re.match('.*((Calculate|calculate|how|How|what|What|Find|find|True or false).*)$',
            #                                  self.example, flags=re.MULTILINE)[1]
            self.overall_question = re.match('.*((([A-Z].*(calculate|how|what|when|find|true or false))|((Calculate|How|What|When|Determine|Find|True or false))).*)', self.example, flags=re.DOTALL)[1]

    def construct_final_action(self, state: GSM8kState,) -> GSM8kAction:
        with io.StringIO() as f:
            f.write(self.prompt["overall_question_prefix"])
            f.write(" " + self.overall_question)
            model_input = f.getvalue()
        output = list(dict.fromkeys([model_input]))[0]
        return output
        

    def get_actions(self, state: GSM8kState, ) -> list[GSM8kAction]:
        with io.StringIO() as f:
            f.write(self.prompt["input"])
            f.write(self.prompt["question_prefix"] + self.example + "\n")
            f.write(self.prompt["answer_prefix"])
            for a in state:
                f.write(a.sub_question + " ")
            if at_depth_limit := self.force_terminating_on_depth_limit and len(state) + 1 >= self.depth_limit:
                f.write(" " + self.prompt["overall_question_prefix"])##seems the model self stop without the limit
            model_input = f.getvalue()

        n_actions = self.n_actions
        temperature = self.temperature
        outputs = []
        print("____________________________")
        print("model_input:",model_input)   
        print("____________________________")
        for idx in range(0, n_actions, self.batch_size):
            n_samples = min(n_actions - idx, self.batch_size)
            outputs += self.base_model.generate([model_input] * n_samples,
                                                hide_input=False,
                                                do_sample=True,
                                                temperature=temperature,
                                                eos_token_id=None).text

        outputs = [output.split(model_input)[1].strip() for output in outputs]
        print(['*']*20)
        print(outputs)
        print(['*']*20)
        # if at_depth_limit:
        #     outputs = [self.prompt["overall_question_prefix"] + ' ' + output for output in outputs]
        # if self.force_overall_question_on_overall_prompt:
        #     for i, output in enumerate(outputs):
        #         if self.prompt["overall_question_prefix"] in output:
        #             outputs[i] = self.prompt["overall_question_prefix"] + ' ' + self.overall_question
        # if self.force_overall_prompt_on_overall_question:
        #     for i, output in enumerate(outputs):
        #         if self.overall_question.lower() == output.lower():
        #             outputs[i] = self.prompt["overall_question_prefix"] + ' ' + self.overall_question
        
        if at_depth_limit:
            for k in range(len(outputs)):
                # print(model_input)
                # print(outputs)
                if 'Q:' in outputs[k]:
                    outputs[k] = outputs[k].split('Q:')[0].rstrip('\n')
                if '\n' in outputs[k]:
                    outputs[k] = outputs[k].split('\n')[0].rstrip('.') + '.'
                else:
                    outputs[k] = outputs[k].split('. ',1)[0].rstrip('.') + '.'#only keep one sent for one action 
                if 'The answer is' in outputs[k]:
                    outputs[k] = outputs[k].split('The answer is')[1].rstrip('\n').rstrip('.') + '.'
                outputs[k] = self.prompt["overall_question_prefix"] + ' ' + outputs[k]
        else:
            for k in range(len(outputs)):
                outputs[k] = outputs[k].rstrip('\n')
                if 'Q:' in outputs[k]:
                    outputs[k] = outputs[k].split('Q:')[0].rstrip('\n')
                if '\n' in outputs[k]:
                    outputs[k] = outputs[k].split('\n')[0].rstrip('.') + '.'
                else:
                    tmp = outputs[k].split('. ',1)[0].rstrip('.') + '.'#only keep one sent for one action
                    if len(tmp) < 5:#like 1. 2.
                        tmp = '. '.join(outputs[k].split('. ',2)[0:2]).rstrip('.') + '.'
                    outputs[k] = tmp
        # set does not guarantee order, but dict does guarantee
        # we cannot use set here because torch.distributed in LLaMA requires the same order across all processes
        # print('former:',len(outputs))
        outputs = list(dict.fromkeys(outputs))
        # print('latter:',len(outputs))
        return outputs

    def fast_reward(self, state: GSM8kState, action: GSM8kAction) -> tuple[float, dict]:
        with io.StringIO() as f:
            f.write(self.useful_prompt["input"])
            f.write(self.useful_prompt["question_prefix"] + self.example + "\n")
            for idx, (q, _, _) in enumerate(state):
                f.write(self.useful_prompt["subquestion_prefix"].format(idx + 1) + " " + q + "\n")
            f.write(self.useful_prompt["new_subquestion_prefix"].format(len(state) + 1) + " " + action + "\n")
            f.write(self.useful_prompt["useful_prefix"])
            model_input = f.getvalue()

        logits = self.base_model.get_next_token_logits(model_input, ["Yes", "No"])[0]
        probs = np.exp(logits) / np.sum(np.exp(logits))
        useful_prob = probs[0]
        fast_reward, _ = self.calculate_reward(useful_prob)
        return fast_reward, {'r_useful': useful_prob}

    def calculate_reward(self, r_useful, r_conf=None):
        if r_conf is None:
            r_conf = self.reward_confidence_default
        return r_useful ** self.reward_alpha * r_conf ** (1 - self.reward_alpha), {'r_useful': r_useful,
                                                                                   'r_conf': r_conf}

    def reward(self, state: GSM8kState, action: GSM8kAction,
               r_useful: float = None,
               confidence: float = None) -> tuple[float, dict]:
        assert r_useful is not None, "useful_reward is required to calculate reward in this search config, consider passing it in fast_reward"
        assert confidence is not None, "confidence is required to calculate reward in this search config, consider passing it in world model's step"
        return self.calculate_reward(r_useful, confidence)
