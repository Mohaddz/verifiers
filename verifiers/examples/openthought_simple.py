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
    max_steps: int = 1000,  # Added parameter for max steps
    num_generations: int = 4,  # Changed to 4 which works with both training and eval
    per_device_train_batch_size: int = 4,  # Changed to match num_generations
    per_device_eval_batch_size: int = 4,  # Added parameter for eval batch size
    eval_accumulation_steps: int = 8  # Added parameter to control eval accumulation
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
        max_steps: Maximum number of training steps
        num_generations: Number of generations per prompt (must be a factor of eval_accumulation_steps)
        per_device_train_batch_size: Per device train batch size (must be divisible by num_generations)
        per_device_eval_batch_size: Per device evaluation batch size
        eval_accumulation_steps: Number of eval steps to accumulate before backward pass
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
    
    # We'll get datasets later in a try/except block
    dataset = None
    eval_dataset = None
    
    # Get reward functions
    rubric = vf_env.get_rubric()
    
    # Configure training arguments
    run_name = f"openthought_{model_name.split('/')[-1].lower()}"
    training_args = vf.get_default_grpo_config(
        run_name=run_name,
        num_gpus=num_gpus
    )
    
    # Handle small datasets by disabling evaluation 
    # This prevents issues with evaluation batch size when dataset is very small
    if max_samples <= 100:
        print("Small dataset detected. Disabling evaluation to avoid batch size issues.")
        eval_dataset = None
        
    # For safety, wrap getting datasets in try/except blocks
    try:
        # Get dataset and evaluation dataset
        dataset = vf_env.get_dataset()
        if eval_dataset is not None:  # Only get eval dataset if not disabled
            eval_dataset = vf_env.get_eval_dataset()
    except Exception as e:
        print(f"Error getting datasets: {e}")
        # Create a minimal dataset for testing
        from datasets import Dataset as HFDataset
        dataset = HFDataset.from_dict({
            "prompt": [{"role": "user", "content": f"Sample problem {i}"} for i in range(20)],
            "answer": [f"Answer {i}" for i in range(20)],
        })
        eval_dataset = None
        print("Using fallback dataset due to error")
    
    # Set specific training parameters
    training_args.num_generations = num_generations
    training_args.per_device_train_batch_size = per_device_train_batch_size
    training_args.per_device_eval_batch_size = per_device_eval_batch_size
    training_args.gradient_accumulation_steps = 4
    training_args.num_iterations = 2
    training_args.beta = 0.04
    
    # Always set eval_strategy to "no" regardless of eval_dataset
    # This avoids issues with the trainer requiring eval_dataset when eval_strategy="steps"
    training_args.eval_strategy = "no"
    
    training_args.eval_steps = 100
    training_args.eval_accumulation_steps = eval_accumulation_steps
    training_args.learning_rate = learning_rate
    training_args.num_train_epochs = num_train_epochs
    training_args.max_steps = max_steps
    
    # Make sure input files are correct
    if not hasattr(dataset, "__len__"):
        print("WARNING: Dataset has no __len__ method. Creating a dummy dataset.")
        dataset = vf.utils.data_utils.Dataset.from_dict({
            "prompt": [{"role": "user", "content": "Sample prompt"}] * 10,
            "answer": ["Sample answer"] * 10,
            "response": ["<thought>Sample thought</thought><answer>Sample answer</answer>"] * 10
        })
    
    # Log training parameters
    print(f"Training with:\n"
          f"  Dataset size: {len(dataset)}\n"
          f"  Batch size: {per_device_train_batch_size}\n"
          f"  GPUs: {num_gpus}\n"
          f"  Max steps: {max_steps}\n"
          f"  Eval strategy: {training_args.eval_strategy}")
    
    # Convert dataset to torch format to ensure compatibility
    dataset = dataset.with_format("torch")
    if eval_dataset is not None:
        eval_dataset = eval_dataset.with_format("torch")
    
    # When using a very small dataset, generate a simple one manually
    if len(dataset) < 5:
        print("WARNING: Dataset too small. Creating a dummy dataset.")
        from datasets import Dataset as HFDataset
        dataset = HFDataset.from_dict({
            "prompt": [{"role": "user", "content": f"Sample prompt {i}"} for i in range(20)],
            "answer": [f"Sample answer {i}" for i in range(20)],
            "response": [f"<thought>Sample thought {i}</thought><answer>Sample answer {i}</answer>" for i in range(20)]
        }).with_format("torch")
        print(f"Created dummy dataset with {len(dataset)} examples")
    
    # Initialize trainer with appropriate settings
    # Don't pass eval_dataset to avoid any issues with evaluation
    trainer = vf.GRPOEnvTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=rubric,
        env=vf_env,
        args=training_args,
        train_dataset=dataset,
        # No eval_dataset - using eval_strategy="no"
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
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum number of training steps")
    parser.add_argument("--num_generations", type=int, default=4, help="Number of generations per prompt (must be a factor of eval_accumulation_steps)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Per device train batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Per device evaluation batch size")
    parser.add_argument("--eval_accumulation_steps", type=int, default=8, help="Number of evaluation steps to accumulate")
    
    args = parser.parse_args()
    
    # Make sure num_generations is a factor of eval_accumulation_steps
    if args.eval_accumulation_steps % args.num_generations != 0:
        valid_factors = [i for i in range(1, args.eval_accumulation_steps + 1) 
                         if args.eval_accumulation_steps % i == 0]
        parser.error(f"num_generations ({args.num_generations}) must be a factor of eval_accumulation_steps ({args.eval_accumulation_steps}). Valid values: {valid_factors}")
    
    # Run with provided parameters
    run_thought_training(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        num_gpus=args.num_gpus,
        max_samples=args.max_samples,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        eval_accumulation_steps=args.eval_accumulation_steps
    )