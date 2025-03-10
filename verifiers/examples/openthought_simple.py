"""
Example of training a model on thought-based reasoning with open-thought-114k dataset.
"""
import argparse
import verifiers as vf
from verifiers.prompts.system_prompts import THOUGHT_PROMPT
from verifiers.prompts.few_shots import THOUGHT_FEW_SHOT
from verifiers.envs.thought_env import ThoughtEnv


def run_thought_training(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    dataset_path: str = "open-thoughts/OpenThoughts-114k",
    num_gpus: int = 8,
    max_samples: int = 5000,
    learning_rate: float = 1e-6,
    num_train_epochs: int = 3,
    num_generations: int = 6,
    per_device_train_batch_size: int = 6
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
        num_generations: Number of generations per prompt
        per_device_train_batch_size: Per device train batch size
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
    training_args.num_generations = num_generations
    training_args.per_device_train_batch_size = per_device_train_batch_size
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
    parser = argparse.ArgumentParser(description="Train a model on thought-based reasoning tasks")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Name or path of the model to train")
    parser.add_argument("--dataset_path", type=str, default="open-thoughts/OpenThoughts-114k", help="Path to the dataset")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use for training")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum number of samples to use")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate for training")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--num_generations", type=int, default=6, help="Number of generations per prompt")
    parser.add_argument("--per_device_train_batch_size", type=int, default=6, help="Per device train batch size")
    
    args = parser.parse_args()
    
    # Run with provided parameters
    run_thought_training(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        num_gpus=args.num_gpus,
        max_samples=args.max_samples,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_train_batch_size
    )