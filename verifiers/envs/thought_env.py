from typing import List, Dict, Any, Optional
from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc

from verifiers.envs.multistep_env import MultiStepEnv
from verifiers.parsers.xml_parser import XMLParser
from verifiers.prompts.system_prompts import THOUGHT_PROMPT
from verifiers.prompts.few_shots import THOUGHT_FEW_SHOT
from verifiers.rubrics.thought_rubric import ThoughtRubric
from verifiers.utils.data_utils import preprocess_thought_dataset


class ThoughtEnv(MultiStepEnv):
    """Environment for training models on thought-based reasoning."""
    
    def __init__(self,
                 dataset_path: str = "open-thoughts/OpenThoughts-114k",
                 system_prompt: str = THOUGHT_PROMPT,
                 few_shot: List[Dict[str, str]] = THOUGHT_FEW_SHOT[0],
                 sampling_args: Dict[str, Any] = {
                     "stop": ["</thought>", "</answer>"],
                     "include_stop_str_in_output": True
                 },
                 mask_env_response: bool = True,
                 max_steps: int = 3,
                 max_samples: int = 5000,
                 random_seed: int = 42,
                 **kwargs):
        super().__init__(
            system_prompt=system_prompt,
            few_shot=few_shot,
            mask_env_response=mask_env_response,
            sampling_args=sampling_args,
            **kwargs
        )
        self.dataset_path = dataset_path
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.dataset = None
        self.eval_dataset = None
        self.max_steps = max_steps
        self.parser = XMLParser(fields=["thought", "answer"])
        self.rubric = ThoughtRubric(parser=self.parser)
    
    def get_dataset(self, **kwargs: Any) -> Dataset:
        """Get the preprocessed training dataset."""
        if self.dataset is None:
            self.dataset = preprocess_thought_dataset(
                dataset_path=self.dataset_path,
                split="train",
                system_prompt=self.system_prompt,
                few_shot=self.few_shot,
                max_samples=self.max_samples,
                random_seed=self.random_seed
            )
            
            # Verify dataset has a proper length method
            if not hasattr(self.dataset, '__len__'):
                raise ValueError("Dataset doesn't have a __len__ method. Please fix your dataset implementation.")
                
            # Log dataset size
            print(f"Training dataset size: {len(self.dataset)} examples")
            
        return self.dataset
    
    def get_eval_dataset(self, **kwargs: Any) -> Dataset:
        """Get the preprocessed evaluation dataset."""
        if self.eval_dataset is None:
            # Use a fixed proportion of the train split for evaluation
            eval_samples = min(1000, self.max_samples // 5)
            
            # Use a different random seed for evaluation sampling
            eval_seed = self.random_seed + 42
            
            # Get evaluation dataset from train split
            self.eval_dataset = preprocess_thought_dataset(
                dataset_path=self.dataset_path,
                split="train",  # Use train split since there's no test split
                system_prompt=self.system_prompt,
                few_shot=self.few_shot,
                max_samples=eval_samples,
                random_seed=eval_seed  # Different seed ensures no overlap with training data
            )
            
            # Verify dataset has a proper length method
            if not hasattr(self.eval_dataset, '__len__'):
                raise ValueError("Eval dataset doesn't have a __len__ method. Please fix your dataset implementation.")
                
            # Log dataset size
            print(f"Evaluation dataset size: {len(self.eval_dataset)} examples")
        
        return self.eval_dataset
    
    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        """Get reward functions for evaluation."""
        return self.rubric.get_reward_funcs()
    
    def is_completed(self, messages: List[Dict[str, str]], **kwargs: Any) -> bool:
        """Check if the reasoning task is completed."""
        try:
            if not messages or messages[-1]["role"] != "assistant":
                return False
                
            parsed = self.parser.parse(messages[-1]["content"])
            # Check if both thought and answer are present
            has_thought = hasattr(parsed, 'thought') and parsed.thought is not None
            has_answer = hasattr(parsed, 'answer') and parsed.answer is not None
            return has_thought and has_answer
        except Exception:
            return False
    
    def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
        """Generate environment response to guide the model."""
        if not messages or messages[-1]["role"] != "assistant":
            return {"role": "user", "content": "Please provide your reasoning."}
            
        try:
            parsed = self.parser.parse(messages[-1]["content"])
            has_thought = hasattr(parsed, 'thought') and parsed.thought is not None
            has_answer = hasattr(parsed, 'answer') and parsed.answer is not None
            
            if has_thought and not has_answer:
                return {"role": "user", "content": "Continue your reasoning and provide your final answer in <answer> tags."}
            elif not has_thought and has_answer:
                return {"role": "user", "content": "Please show your reasoning in <thought> tags before giving your answer."}
            elif not has_thought and not has_answer:
                return {"role": "user", "content": "Please provide your reasoning in <thought> tags and your answer in <answer> tags."}
        except Exception:
            pass
            
        return {"role": "user", "content": "Please organize your response with reasoning in <thought> tags and final answer in <answer> tags."}