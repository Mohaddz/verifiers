"""
Example of training a model on thought-based reasoning with open-thought-114k dataset.
"""
import verifiers as vf
from verifiers.prompts.system_prompts import THOUGHT_PROMPT
from verifiers.prompts.few_shots import THOUGHT_FEW_SHOT
from verifiers.envs.thought_env import ThoughtEnv


def run_thought_training(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    dataset_path: str = "openai/open-thought-114k",
    num_gpus: int = 8,
    max_samples: int = 5000,
    learning_rate: float = 1e-6,
    num_train_epochs: int = 3
):
    """
    Train a model on thought-based reasoning tasks.
    
    Args:
        model_name: Name or path of the model to train
        dataset_path: Path to the dataset
        num_gpus: Number of GPUs to use for training
        max_samples: Maximum number of samples to use
        learning_rate: Learning rate for training
        num_train_epochs: Number of training epochs
    """
    # Initialize model and tokenizer
    model, tokenizer = vf.get_model_and_tokenizer(model_name)
    
    # Initialize environment
    vf_env = ThoughtEnv(
        dataset_path=dataset_path,
        max_samples=max_samples,
        system_prompt=THOUGHT_PROMPT,
        few_shot=THOUGHT_FEW_SHOT[0]
    )
    
    # Get dataset and evaluation dataset
    dataset = vf_env.get_dataset()
    eval_dataset = vf_env.get_eval_dataset()
    
    # Get reward functions
    rubric = vf_env.get_rubric()
    
    # Configure training arguments
    run_name = f"openthought_{model_name.split('/')[-1].lower()}"
    training_args = vf.get_default_grpo_config(
        run_name=run_name,
        num_gpus=num_gpus
    )
    
    # Set specific training parameters
    training_args.num_generations = 7
    training_args.per_device_train_batch_size = 6
    training_args.gradient_accumulation_steps = 4
    training_args.num_iterations = 2
    training_args.beta = 0.04
    training_args.eval_strategy = "steps"
    training_args.eval_steps = 100
    training_args.eval_accumulation_steps = 8
    training_args.learning_rate = learning_rate
    training_args.num_train_epochs = num_train_epochs
    
    # Initialize trainer
    trainer = vf.GRPOEnvTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=rubric,
        env=vf_env,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset
    )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    # Run with default parameters
    run_thought_training(max_samples=1000)  # Using a smaller sample for testing