from typing import List, Dict, Any, Optional
import re

from verifiers.parsers.xml_parser import XMLParser
from verifiers.rubrics.rubric import Rubric


class ThoughtRubric(Rubric):
    """Rubric for evaluating thought-based reasoning responses."""
    
    def __init__(self, parser: Optional[XMLParser] = None, **kwargs):
        super().__init__(**kwargs)
        self.parser = parser or XMLParser(fields=["thought", "answer"])
        
        self.reward_funcs = [
            self.exact_answer_reward_func,
            self.thought_quality_reward_func,
            self.thought_completeness_reward_func,
            self.parser.get_xml_reward_func(),
            self.parser.get_format_reward_func()
        ]
        
        self.reward_weights = [0.35, 0.25, 0.2, 0.1, 0.1]
    
    def thought_quality_reward_func(self, completions, **kwargs) -> List[float]:
        """Evaluate the quality of reasoning in the thought section."""
        
        def evaluate_thought_quality(trajectory):
            model_messages = self.get_assistant_messages(trajectory)
            if not model_messages:
                return 0.0
                
            # Get the most recent message
            last_message = model_messages[-1]
            
            try:
                parsed = self.parser.parse(last_message['content'])
                if not hasattr(parsed, 'thought') or parsed.thought is None:
                    return 0.0
                    
                thought = parsed.thought.strip()
                
                # Skip very short thoughts
                if len(thought) < 10:
                    return 0.0
                
                # Basic quality indicators
                quality_score = 0.0
                
                # Length-based score (up to 0.3)
                word_count = len(thought.split())
                if word_count >= 100:
                    quality_score += 0.3
                elif word_count >= 50:
                    quality_score += 0.2
                elif word_count >= 20:
                    quality_score += 0.1
                
                # Check for step-by-step reasoning (up to 0.4)
                steps_indicators = ["step", "first", "second", "third", "then", "next", "finally", "lastly"]
                steps_found = 0
                for indicator in steps_indicators:
                    if re.search(r'\b' + indicator + r'\b', thought.lower()):
                        steps_found += 1
                
                if steps_found >= 3:
                    quality_score += 0.4
                elif steps_found >= 1:
                    quality_score += 0.2
                
                # Check for calculations and numeric work (up to 0.3)
                has_calculations = bool(re.search(r'[\d\+\-\*\/\=][\d\s\+\-\*\/\=]+[\d\+\-\*\/\=]', thought))
                if has_calculations:
                    quality_score += 0.3
                
                return quality_score
            except Exception:
                return 0.0
        
        return [evaluate_thought_quality(c) for c in completions]
    
    def thought_completeness_reward_func(self, completions, **kwargs) -> List[float]:
        """Evaluate whether the reasoning is complete and leads to the answer."""
        
        def evaluate_completeness(trajectory):
            model_messages = self.get_assistant_messages(trajectory)
            if not model_messages:
                return 0.0
                
            # Get the most recent message
            last_message = model_messages[-1]
            
            try:
                parsed = self.parser.parse(last_message['content'])
                if not hasattr(parsed, 'thought') or not hasattr(parsed, 'answer'):
                    return 0.0
                    
                if parsed.thought is None or parsed.answer is None:
                    return 0.0
                    
                thought = parsed.thought.strip()
                answer = parsed.answer.strip()
                
                # Skip cases with very short thoughts or answers
                if len(thought) < 10 or len(answer) < 1:
                    return 0.0
                
                # Check if the answer logically follows from the thought
                last_thought_sentence = thought.split('.')[-2] if len(thought.split('.')) > 1 else thought
                
                # Connection between thought and answer (up to 1.0)
                completeness_score = 0.0
                
                # Check if answer contains numbers found in the last part of thought
                thought_numbers = re.findall(r'\d+(?:\.\d+)?', last_thought_sentence)
                answer_numbers = re.findall(r'\d+(?:\.\d+)?', answer)
                
                if thought_numbers and any(num in answer_numbers for num in thought_numbers):
                    completeness_score += 0.5
                
                # Check for conclusion words near the end of thought
                conclusion_indicators = ["therefore", "thus", "so", "hence", "conclude", "result"]
                has_conclusion = any(indicator in last_thought_sentence.lower() for indicator in conclusion_indicators)
                
                if has_conclusion:
                    completeness_score += 0.5
                
                return completeness_score
            except Exception:
                return 0.0
        
        return [evaluate_completeness(c) for c in completions]
    
    def exact_answer_reward_func(self, completions, **kwargs) -> List[float]:
        """Evaluate whether the final answer matches the expected answer."""
        # This is needed even though we've defined thought_quality, because the default
        # implementation will still be called during evaluation.
        
        def eval_answer(trajectory):
            if not hasattr(trajectory, 'prompt'):
                return 0.0
                
            target = trajectory.prompt.get('answer', None)
            if target is None:
                return 0.0
                
            model_messages = self.get_assistant_messages(trajectory)
            if not model_messages:
                return 0.0
                
            # Get the most recent message
            last_message = model_messages[-1]
            
            try:
                parsed = self.parser.parse(last_message['content'])
                if not hasattr(parsed, 'answer') or parsed.answer is None:
                    return 0.0
                    
                answer = parsed.answer.strip()
                
                # Simple exact match evaluation (consider more sophisticated evaluation in practice)
                if answer.lower() == target.lower():
                    return 1.0
                elif target.lower() in answer.lower() or answer.lower() in target.lower():
                    return 0.5
                else:
                    return 0.0
            except Exception:
                return 0.0
                
        return [eval_answer(c) for c in completions]