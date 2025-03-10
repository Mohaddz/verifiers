import random
import json
import re
from typing import List, Dict, Any, Optional, Union

from datasets import Dataset, load_dataset # type: ignore

def extract_boxed_answer(text: str) -> str | None:
    def find_matching_brace(s: str, start: int) -> int:
        count = 1
        i = start
        while i < len(s) and count > 0:
            if s[i] == '{':
                count += 1
            elif s[i] == '}':
                count -= 1
            i += 1
        return i - 1 if count == 0 else -1

    # Find \boxed{
    boxed_start = text.find('\\boxed{')
    if boxed_start == -1:
        return text
    # Find the content between the braces
    content_start = boxed_start + 7  # len('\\boxed{')
    closing_brace = find_matching_brace(text, content_start)
    
    if closing_brace == -1:
        return text
    
    return text[content_start:closing_brace]

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def format_prompt(prompt: str,
                  system_prompt: str | None = None,
                  few_shot: List[Dict[str, str]] | None = None,
                  fewshot_prob: float = 1.0) -> List[Dict[str, str]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if few_shot and random.random() < fewshot_prob:
        messages.extend(few_shot)
    messages.append({"role": "user", "content": prompt})
    return messages

def preprocess_dataset(dataset_name: str = "gsm8k", 
                       split: str = "train",
                       system_prompt: str | None = None,
                       few_shot: List[Dict[str, str]] | None = None,
                       fewshot_prob: float = 1.0) -> Dataset:
    if dataset_name == "gsm8k":
        dataset: Dataset = load_dataset("openai/gsm8k", "main")[split] # type: ignore
        dataset = dataset.map(lambda x: {
            "prompt": format_prompt(x["question"], system_prompt, few_shot, fewshot_prob),
            "answer": extract_hash_answer(x["answer"])
        })
        return dataset
    elif dataset_name == "math":
        dataset: Dataset = load_dataset("chiayewken/competition_math")[split] # type: ignore
        dataset = dataset.map(lambda x: {
            "prompt": format_prompt(x["problem"], system_prompt, few_shot, fewshot_prob),
            "answer": extract_boxed_answer(x["solution"])
        })
        return dataset
    elif dataset_name == "openbookqa":
        dataset: Dataset = load_dataset("allenai/openbookqa", "main")[split] # type: ignore
        
        def format_question(example):
            choices_texts = example['choices']['text']
            choices_labels = example['choices']['label']
            
            formatted_choices = []
            for i in range(len(choices_labels)):
                formatted_choices.append(f"{choices_labels[i]}. {choices_texts[i]}")
            
            question = f"Question: {example['question_stem']}\n\nChoices:\n" + "\n".join(formatted_choices)
            return question
        
        dataset = dataset.map(lambda x: {
            "prompt": format_prompt(format_question(x), str(system_prompt) + "\n\nReturn only the letter of the correct answer.", few_shot, fewshot_prob),
            "answer": x["answerKey"]
        })
        return dataset
    else:
        raise ValueError(f"Dataset {dataset_name} not supported for preprocess_dataset.")

def preprocess_thought_dataset(
        dataset_path: str = "openai/open-thought-114k",
        split: str = "train",
        system_prompt: Optional[str] = None,
        few_shot: Optional[List[Dict[str, str]]] = None,
        max_samples: int = 5000,
        random_seed: int = 42) -> Dataset:
    """
    Load and preprocess the open-thought dataset for thought-based reasoning training.
    
    Args:
        dataset_path: Path or name of the dataset
        split: Dataset split to use (train, test, validation)
        system_prompt: Optional system prompt to include
        few_shot: Optional few-shot examples to include
        max_samples: Maximum number of samples to include
        random_seed: Random seed for reproducible sampling
        
    Returns:
        Preprocessed dataset
    """
    # Load dataset
    try:
        dataset = load_dataset(dataset_path, split=split)
    except Exception as e:
        print(f"Error loading dataset {dataset_path}: {e}")
        return Dataset.from_dict({"prompt": [], "answer": []})
    
    # Sample a subset if needed
    if max_samples is not None and max_samples < len(dataset):
        # Set seed for reproducibility
        random.seed(random_seed)
        sample_indices = random.sample(range(len(dataset)), max_samples)
        dataset = dataset.select(sample_indices)
    
    def convert_thought_tags(text: str) -> str:
        """Convert open-thought tags to the XML format used in the codebase."""
        # Replace thought tags
        text = text.replace("<|begin_of_thought|>", "<thought>")
        text = text.replace("<|end_of_thought|>", "</thought>")
        
        # Replace solution tags
        text = text.replace("<|begin_of_solution|>", "<answer>")
        text = text.replace("<|end_of_solution|>", "</answer>")
        
        return text
    
    def extract_answer(example: Dict[str, Any]) -> Dict[str, Any]:
        """Extract answer from assistant response."""
        try:
            # Get the conversation from the example
            conversation = example.get("conversation", [])
            
            # Extract user query from first message
            user_query = ""
            if conversation and len(conversation) > 0:
                user_query = conversation[0].get("value", "")
            
            # Extract assistant response from second message
            assistant_response = ""
            if conversation and len(conversation) > 1:
                assistant_response = conversation[1].get("value", "")
            
            # Convert tags in assistant response
            formatted_response = convert_thought_tags(assistant_response)
            
            # Extract answer part
            answer = ""
            answer_match = re.search(r"<answer>(.*?)</answer>", formatted_response, re.DOTALL)
            if answer_match:
                answer = answer_match.group(1).strip()
            
            # Format prompt with existing format_prompt function
            formatted_prompt = format_prompt(user_query, system_prompt, few_shot)
            
            return {
                "prompt": formatted_prompt,
                "response": formatted_response,
                "answer": answer
            }
        except Exception as e:
            print(f"Error processing example: {e}")
            return {
                "prompt": [],
                "response": "",
                "answer": ""
            }
    
    # Process each example in the dataset
    processed_dataset = dataset.map(extract_answer)
    
    # Filter out examples without prompts or answers
    processed_dataset = processed_dataset.filter(lambda x: x["prompt"] and x["answer"])
    
    return processed_dataset