import argparse
import functools
import gc
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from datasets import load_dataset
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM, CompletionOutput, RequestOutput, SamplingParams, TokensPrompt

# FSDP Imports
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    CPUOffload,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp import ShardingStrategy

import wandb
from utils import (
    clean_up_checkpoints,
    close_to_zero,
    compute_token_log_probs,
    dump_episodes,
    evaluate_on_test_set,
    find_last_checkpoint,
    fix_oov_logits_processor,
    initialize_training_process_group,
    ensure_master_addr_port,
    move_model_to_vllm,
    prepare_model_inputs,
)
from math_equivalence import extract_boxed_answer

os.environ["VLLM_USE_V1"] = "0"

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("[%(levelname)s|%(filename)s:%(lineno)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

arg_parser = argparse.ArgumentParser(description="Train R1 model with PPO")
arg_parser.add_argument("--kl_coeff", type=float, default=0.001, help="KL coefficient for PPO")
arg_parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
arg_parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B", help="Model name/path")
arg_parser.add_argument("--per_device_batch_size", type=int, default=8, help="Per device batch size")
arg_parser.add_argument("--max_response_tokens", type=int, default=512, help="Max response tokens")
arg_parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate for training")
arg_parser.add_argument("--debug", action="store_true", help="Debug mode")
arg_parser.add_argument("--algorithm", type=str, choices=["grpo", "vineppo"], default="grpo", help="Algorithm to use")
arg_parser.add_argument("--run_id", type=str, default=None, help="Run ID")
arg_parser.add_argument("--nproc", type=int, default=1, help="Number of processes (data parallelism) to use")
arg_parser.add_argument("--run_dir", type=str, default=None, help="Base directory for run artifacts")
arg_parser.add_argument("--save_steps", type=int, default=50, help="Save frequency in steps")
arg_parser.add_argument("--test_freq", type=int, default=25, help="Test frequency in steps")
arg_parser.add_argument("--data_path", type=str, default=None, help="Path to the dataset")
arg_parser.add_argument("--data_size", type=int, default=None, help="Size of the dataset to use")
arg_parser.add_argument("--max_iter", type=int, default=1000, help="Maximum number of iterations")

# Load and process dataset
def preprocess_example(example: Dict[str, Any], tokenizer: AutoTokenizer):
    prefix = example["prompt"]
    input_ids = tokenizer.apply_chat_template(prefix, tokenize=True, add_generation_prompt=True)['input_ids']
    
    # Force cast to standard Python integers to avoid Arrow/Dataset typing issues
    # input_ids = [int(x) for x in input_ids] 
    
    prompt = tokenizer.decode(input_ids, skip_special_tokens=False)
    return {"prompt": prompt, "input_ids": input_ids}

def equation_reward_func(completion: str, target: int) -> float:
    try:
        predicted_answer = extract_boxed_answer(completion)
        result = int(predicted_answer)

        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
        else:
            return 0.0
    except Exception:
        return 0.0


def compute_reward(completion: str, sample: Dict[str, Any], EOS_TOKEN: str) -> Tuple[float, Dict[str, float]]:
    target = sample["reward_model"]["ground_truth"]
    equation_reward = equation_reward_func(completion=completion, target=target)
    reward = equation_reward
    metrics = {
        "equation_reward": equation_reward,
    }
    return reward, metrics


def create_training_episodes(
    *,
    samples: List[Dict[str, Any]] = None,
    all_generations: List[List[int]] = None,
    all_finish_reasons: List[str] = None,
    tokenizer: AutoTokenizer = None,
    EOS_TOKEN: str = None,
    GENERATIONS_PER_SAMPLE: int = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    assert len(all_generations) == len(all_finish_reasons)
    assert len(all_generations) == len(samples) * GENERATIONS_PER_SAMPLE

    groups = [
        list(range(i, i + GENERATIONS_PER_SAMPLE)) for i in range(0, len(all_generations), GENERATIONS_PER_SAMPLE)
    ]

    all_query_token_ids, all_responses_token_ids, all_advantages = [], [], []

    stats = {
        "response_lengths": [],
        "rewards": [],
        "non_stop_rate": [],
    }

    for sample, group_indices in zip(samples, groups):
        response_token_ids = [all_generations[i] for i in group_indices]
        finish_reasons = [all_finish_reasons[i] for i in group_indices]
        responses = tokenizer.batch_decode(response_token_ids, skip_special_tokens=False)
        rewards_and_metrics = [compute_reward(resp, sample, EOS_TOKEN) for resp in responses]
        rewards, reward_metrics = zip(*rewards_and_metrics)

        rewards = np.array(rewards)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

        per_token_advantages = [[adv] * len(resp) for adv, resp in zip(advantages, response_token_ids)]

        all_query_token_ids.extend([sample["input_ids"]] * GENERATIONS_PER_SAMPLE)
        all_responses_token_ids.extend(response_token_ids)
        all_advantages.extend(per_token_advantages)

        stats["rewards"].extend(rewards)
        stats["non_stop_rate"].extend([fr != "stop" for fr in finish_reasons])
        stats["response_lengths"].extend([len(ids) for ids in response_token_ids])
        for rm in reward_metrics:
            for k, v in rm.items():
                stats.setdefault(f"reward_metrics/{k}", []).append(v)

    episodes = {
        "all_query_token_ids": all_query_token_ids,
        "all_response_token_ids": all_responses_token_ids,
        "all_advantages": all_advantages,
    }

    return episodes, stats

def compute_pg_loss(
    policy_model: Union[FSDP, PreTrainedModel],
    batch: Dict[str, torch.Tensor],
    total_response_len: torch.Tensor,
    TEMPERATURE: float,
    KL_COEFFICIENT: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    
    input_ids = batch["input_ids"] 
    attention_mask = batch["attention_mask"]  
    labels = batch["labels"] 
    labels_mask = batch["labels_mask"]  
    advantages = batch["advantages"] 
    ref_logps = batch["ref_log_probs"] 

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "labels_mask": labels_mask,
    }

    logps = compute_token_log_probs(policy_model, model_inputs, TEMPERATURE)  
    labels_mask = labels_mask[..., 1:].to(logps.dtype)  

    ref_logratio = ref_logps - logps
    kl_penalty = torch.exp(ref_logratio) - 1 - ref_logratio 
    kl_penalty = kl_penalty * labels_mask  

    with torch.no_grad():
        entropy = -logps.sum() / labels_mask.sum()
        zero_advantages = close_to_zero(advantages[..., 1:], labels_mask) 

    policy_loss = -logps * advantages[..., 1:] 
    policy_loss = policy_loss * labels_mask  

    loss = (policy_loss + KL_COEFFICIENT * kl_penalty).sum() / total_response_len

    metrics = {
        "policy_loss": policy_loss.sum().item() / total_response_len.item(),
        "kl_penalty": kl_penalty.sum().item() / total_response_len.item(),
        "entropy": entropy.item(),
        "zero_advantages_ratio": zero_advantages.item() / total_response_len.item(),
    }

    return loss, metrics


def main(rank: int):
    args = arg_parser.parse_args()

    nproc = args.nproc
    initialize_training_process_group(rank, nproc)
    curr_cuda_device = torch.device("cuda")
    # torch.cuda.set_device(curr_cuda_device)

    if dist.get_rank() != 0:
        logger.setLevel(logging.ERROR)

    if args.debug and nproc == 1:
        import debugpy
        debugpy.listen(5678)
        logger.info("Waiting for debugger to attach...")
        debugpy.wait_for_client()
        logger.info("Debugger attached")

    ############################################
    # Hyperparameters
    ############################################

    MODEL_NAME = args.model_name
    ENABLE_SLEEP_MODE = True

    NUM_ITERATIONS = args.max_iter
    EPISODES_PER_ITERATION = 64
    EPISODES_PER_ITERATION_PER_RANK = EPISODES_PER_ITERATION // dist.get_world_size()
    GENERATIONS_PER_SAMPLE = 4
    KL_COEFFICIENT = args.kl_coeff

    PER_DEVICE_BATCH_SIZE = args.per_device_batch_size
    LEARNING_RATE = args.learning_rate

    MAX_RESPONSE_TOKENS = args.max_response_tokens
    TEMPERATURE = args.temperature
    TOP_P = 0.999  
    TOP_K = -1  

    dist.barrier(device_ids=[torch.cuda.current_device()])

    model_name_short = MODEL_NAME.split("/")[-1]
    if args.run_id is None:
        RUN_NAME = f"{model_name_short}_temp{TEMPERATURE}_kl{KL_COEFFICIENT}_lr{LEARNING_RATE}_al{args.algorithm}"
    else:
        RUN_NAME = args.run_id
    if args.run_dir:
        EXP_DIR = Path(args.run_dir) / RUN_NAME
    else:
        raise ValueError("args.run_dir must be specified.")
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Logs and Checkpoints will be saved to: {EXP_DIR}")

    ############################################
    # Prompts and Dataset
    ############################################

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    EOS_TOKEN_ID = tokenizer.eos_token_id
    EOS_TOKEN = tokenizer.convert_ids_to_tokens(EOS_TOKEN_ID)
    tokenizer.chat_template = """{% for message in messages %}
{% if message['role'] == 'system' %}
{{ message['content'] }}
{% elif message['role'] == 'user' %}
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Solve the following math problem, and put your final answer within \\boxed{}.

### Input:
{{ message['content'] }}

### Response:
{% elif message['role'] == 'assistant' %}
{{ message['content'] }}
{% endif %}
{% endfor %}"""

    dataset = load_dataset("parquet", 
                            data_files=args.data_path)['train']
    if dist.get_rank() != 0:
        dist.barrier(device_ids=[torch.cuda.current_device()])
    if args.data_size is not None:
        dataset = dataset.select(range(args.data_size))
    dataset = dataset.map(
        preprocess_example,
        num_proc=6,
        fn_kwargs={
            "tokenizer": tokenizer,
        },
        desc="Preprocessing dataset",
    )
    print(dataset[0])
    if dist.get_rank() == 0:
        dist.barrier(device_ids=[torch.cuda.current_device()])
    dist.barrier(device_ids=[torch.cuda.current_device()])

    

    train_test_split = dataset.train_test_split(test_size=10, seed=42)
    train_dataset = train_test_split["train"]
    orig_train_dataset_size = len(train_dataset)
    test_dataset = train_test_split["test"]

    train_dataset = train_dataset.shard(num_shards=dist.get_world_size(), index=dist.get_rank())

    logger.info(f"Train dataset size: {orig_train_dataset_size}; each rank will process {len(train_dataset)} samples")
    logger.info(f"Test dataset size: {len(test_dataset)}")

    ############################################
    # Initialize Models (FSDP)
    ############################################

    # Load local base model weights
    policy_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    reference_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    
    # Generic wrapper that works securely without arch-specific imports 
    auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1e6)
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # Initialize FSDP Policies
    policy_model = FSDP(
        policy_model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.NO_SHARD,
    )

    reference_model = FSDP(
        reference_model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        device_id=torch.cuda.current_device(),
        # FSDP will automagically stream parameters to GPU and release them during forward passes
        cpu_offload=CPUOffload(offload_params=True), 
        sharding_strategy=ShardingStrategy.NO_SHARD,
    )

    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        fused=True,
    )

    dist.barrier(device_ids=[torch.cuda.current_device()])

    ############################################
    # Initialize vLLM (Inference) engine
    ############################################

    if dist.get_rank() != 0:
        vllm_logger = logging.getLogger("vllm")
        vllm_logger.setLevel(logging.ERROR)

    print("Rank: ", dist.get_rank(), "Current device: ", torch.cuda.current_device())
    os.environ["RANK"] = str(dist.get_rank())
    os.environ["LOCAL_RANK"] = str(dist.get_rank())
    os.environ["WORLD_SIZE"] = str(dist.get_world_size())
    ensure_master_addr_port()
    
    inference_engine = LLM(
        model=MODEL_NAME,
        skip_tokenizer_init=False,
        gpu_memory_utilization=0.3,
        enable_prefix_caching=True,
        swap_space=4,
        scheduling_policy="fcfs",
        dtype=torch.bfloat16,
        max_model_len=MAX_RESPONSE_TOKENS + 512,
        tensor_parallel_size=1,
        distributed_executor_backend="external_launcher",
        seed=dist.get_rank() // 1,
        max_num_batched_tokens=4096,
        enable_sleep_mode=ENABLE_SLEEP_MODE,
        logprobs_mode="processed_logprobs",
    )

    generation_kwargs = {
        "n": GENERATIONS_PER_SAMPLE,
        "repetition_penalty": 1.0,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "min_p": 0.0,
        "max_tokens": MAX_RESPONSE_TOKENS,
        "logprobs": 0,  
        "detokenize": False,
        "stop_token_ids": [EOS_TOKEN_ID],
    }
    sampling_params = SamplingParams(**generation_kwargs)

    if dist.get_rank() == 0:
        wandb.init(
            project="RLVR-pathstar",
            name=RUN_NAME,
            resume="allow",
            config={
                "model_name": MODEL_NAME,
                "learning_rate": LEARNING_RATE,
                "num_iterations": NUM_ITERATIONS,
                "episodes_per_iteration": EPISODES_PER_ITERATION,
                "rollouts_per_episode": GENERATIONS_PER_SAMPLE,
                "kl_coefficient": KL_COEFFICIENT,
                "temperature": TEMPERATURE,
                "algorithm": args.algorithm,
            },
        )

    sampler_rng = np.random.default_rng(seed=42)
    NUM_SAMPLES_PER_ITERATION = EPISODES_PER_ITERATION_PER_RANK // GENERATIONS_PER_SAMPLE

    begin_iter = 0
    ckpt_path, ckpt_iter = find_last_checkpoint(EXP_DIR)
    
    # Checkpointing restore needs manual loading for standard PyTorch
    if ckpt_path is not None:
        logger.info(f"Resuming from checkpoint {ckpt_path} at iteration {ckpt_iter}")
        # Note: Depending on how find_last_checkpoint is structured, 
        # ensure you load standard PyTorch weights natively if continuing from FSDP runs
        # e.g., policy_model.load_state_dict(torch.load(ckpt_path / "pytorch_model.bin"))
        begin_iter = ckpt_iter + 1

        with FSDP.summon_full_params(policy_model, writeback=False, rank0_only=False):
            move_model_to_vllm(policy_model.module, inference_engine)

        logger.info(f"Skipping {ckpt_iter} rounds of samples")
        for _ in trange(ckpt_iter, disable=dist.get_rank() != 0):
            _ = sampler_rng.choice(len(train_dataset), size=NUM_SAMPLES_PER_ITERATION, replace=False)

    last_loaded_iter = begin_iter
    for iteration in trange(begin_iter, NUM_ITERATIONS):
        logger.info(f"Iteration {iteration}/{NUM_ITERATIONS}")

        if ENABLE_SLEEP_MODE:
            torch.cuda.empty_cache()
            inference_engine.wake_up(tags=['weights'])
            inference_engine.collective_rpc("reload_weights")
        
        if last_loaded_iter != iteration:
            # Gather FSDP params across shards before passing to vLLM
            with FSDP.summon_full_params(policy_model, writeback=False, rank0_only=False):
                move_model_to_vllm(policy_model.module, inference_engine)
            last_loaded_iter = iteration
        
        if ENABLE_SLEEP_MODE:
            inference_engine.wake_up(tags=['kv_cache'])

        metrics = {}

        #########################################################
        # Evaluation
        #########################################################

        eval_stats = None
        if iteration % args.test_freq == 0 and dist.get_rank() == 0:  
            logger.info("Evaluating on eval set...")
            eval_episodes, eval_stats = evaluate_on_test_set(
                inference_engine=inference_engine,
                test_dataset=test_dataset,
                tokenizer=tokenizer,
                eos_token=EOS_TOKEN,
                eval_sampling_params=SamplingParams(
                    temperature=0.3,
                    max_tokens=MAX_RESPONSE_TOKENS,
                    n=1,
                    detokenize=False,
                    stop_token_ids=[EOS_TOKEN_ID],
                ),
                reward_func=lambda completion, sample: compute_reward(completion, sample, EOS_TOKEN),
            )
            eval_episode_table = dump_episodes(
                episodes=eval_episodes,
                episodes_stats=eval_stats,
                exp_dir=EXP_DIR,
                tokenizer=tokenizer,
                iteration=iteration,
                is_eval=True,
            )
            wandb.log({"eval/episodes": eval_episode_table, "iteration": iteration})
        dist.barrier(device_ids=[torch.cuda.current_device()])

        #########################################################
        # Generate Episodes
        #########################################################

        indices = sampler_rng.choice(len(train_dataset), size=NUM_SAMPLES_PER_ITERATION, replace=False)
        samples = train_dataset.select(indices)

        formatted_inputs = [TokensPrompt(prompt_token_ids=samples["input_ids"][i]) for i in range(len(samples))]

        gen_time = time.time()
        outputs = inference_engine.generate(
            formatted_inputs,
            sampling_params=sampling_params,
        )
        all_generations = [list(g.token_ids) for out in outputs for g in out.outputs]
        all_finish_reasons = [g.finish_reason for out in outputs for g in out.outputs]

        logger.info(f"Generated {len(all_generations)} responses")
        logger.info(f"Time taken to generate {len(all_generations)} responses: {time.time() - gen_time} seconds")

        if args.algorithm == "grpo":
            episodes, episodes_stats = create_training_episodes(
                samples=samples,
                all_generations=all_generations,
                all_finish_reasons=all_finish_reasons,
                tokenizer=tokenizer,
                EOS_TOKEN=EOS_TOKEN,
                GENERATIONS_PER_SAMPLE=GENERATIONS_PER_SAMPLE,
            )
        else:
            raise ValueError(f"Invalid algorithm: {args.algorithm}")

        if ENABLE_SLEEP_MODE:
            inference_engine.sleep(level=2)
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        for k, v in episodes_stats.items():
            metrics.setdefault(k, []).extend(v)

        episode_table = dump_episodes(
            episodes=episodes,
            episodes_stats=episodes_stats,
            exp_dir=EXP_DIR,
            tokenizer=tokenizer,
            iteration=iteration,
            do_save=iteration % 10 == 0 or iteration == 0,
        )

        #########################################################
        # Training
        #########################################################

        model_inputs = prepare_model_inputs(
            query_token_ids=episodes["all_query_token_ids"],
            response_token_ids=episodes["all_response_token_ids"],
            advantages=episodes["all_advantages"],
            device=curr_cuda_device,
        )

        reference_model.eval()

        with torch.no_grad():
            ref_log_probs = []
            for i in trange(
                0,
                EPISODES_PER_ITERATION_PER_RANK,
                PER_DEVICE_BATCH_SIZE,
                desc="Computing reference logprobs",
                disable=dist.get_rank() != 0,
            ):
                batch = {k: v[i : i + PER_DEVICE_BATCH_SIZE] for k, v in model_inputs.items()}
                # FSDP CPUOffload will seamlessly handle the to/from GPU swaps during this computation
                ref_log_probs.append(
                    compute_token_log_probs(
                        model=reference_model,
                        inputs=batch,
                        temperature=TEMPERATURE,
                    )
                )
            ref_log_probs = torch.cat(ref_log_probs)
            model_inputs["ref_log_probs"] = ref_log_probs
            del ref_log_probs

        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        policy_model.train()
        total_response_len = (model_inputs["labels"] != -100).sum()
        train_time = time.time()

        # Zero gradients explicitly prior to our accumulation loop
        optimizer.zero_grad()

        for i in trange(
            0,
            EPISODES_PER_ITERATION_PER_RANK,
            PER_DEVICE_BATCH_SIZE,
            desc="Gradient Accumulation",
            disable=dist.get_rank() != 0,
        ):
            batch = {k: v[i : i + PER_DEVICE_BATCH_SIZE] for k, v in model_inputs.items()}

            loss, loss_metrics = compute_pg_loss(
                policy_model=policy_model,
                batch=batch,
                total_response_len=total_response_len,
                TEMPERATURE=TEMPERATURE,
                KL_COEFFICIENT=KL_COEFFICIENT,
            )

            metrics.setdefault("loss", []).append(loss.item())
            for k, v in loss_metrics.items():
                metrics.setdefault(k, []).append(v.item() if isinstance(v, torch.Tensor) else v)

            # Manual backward pass for gradient accumulation
            loss.backward()
            del loss, loss_metrics

        # FSDP Gradient Clipping & Step
        grad_norm = policy_model.clip_grad_norm_(1.0)
        optimizer.step()
        
        metrics.setdefault("grad_norm", []).append(grad_norm.item() if grad_norm is not None else 0.0)

        logger.info(f"Time taken to train: {time.time() - train_time} seconds")

        #########################################################
        # Update inference engine weights
        #########################################################

        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        #########################################################
        # Log metrics
        #########################################################

        if dist.get_rank() == 0:
            train_metrics = {k: np.mean(v) for k, v in metrics.items() if None not in v}
            # Accessing LR natively through the optimizer groups
            train_metrics["learning_rate"] = optimizer.param_groups[0]["lr"] 
            logs = {
                "iteration": iteration,
                f"train/episodes": episode_table,
                **{f"train/{k}": v for k, v in train_metrics.items()},
            }
            if eval_stats is not None:
                logs.update({f"eval/{k}": np.mean(v) for k, v in eval_stats.items()})
            wandb.log(logs)

            selected_keys = [
                "train/kl_penalty",
                "train/rewards",
                "train/reward_metrics/format_reward",
                "train/reward_metrics/equation_reward",
                "train/response_lengths",
                "eval/rewards",
                "eval/reward_metrics/format_reward",
                "eval/reward_metrics/equation_reward",
            ]
            selected_metrics = {k: float(logs[k]) for k in selected_keys if k in logs}
            logger.info(f"KEY METRICS: {selected_metrics}")

        if iteration % args.save_steps == 0 and iteration != 0:
            ckpt_dir = EXP_DIR / "checkpoints" / f"ckpt_{iteration:06d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Saving FSDP checkpoint as standard HuggingFace Model")
            
            # Configure FSDP to gather all states to rank 0 for saving a standard HF state_dict
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(policy_model, StateDictType.FULL_STATE_DICT, save_policy):
                state_dict = policy_model.state_dict()
                if dist.get_rank() == 0:
                    policy_model.module.save_pretrained(str(ckpt_dir / "hf_model"), state_dict=state_dict)
                    tokenizer.save_pretrained(str(ckpt_dir / "hf_model"))
            
            dist.barrier(device_ids=[torch.cuda.current_device()])

            if dist.get_rank() == 0:
                clean_up_checkpoints(
                    exp_dir=EXP_DIR,
                    keep_every_n_steps=args.save_steps,
                    exclude=[ckpt_dir],
                )
            dist.barrier(device_ids=[torch.cuda.current_device()])

    dist.destroy_process_group()

if __name__ == "__main__":
    args = arg_parser.parse_args()

    n_gpus = torch.cuda.device_count()
    if args.nproc > n_gpus:
        raise ValueError(f"Requested {args.nproc} processes, but only {n_gpus} GPUs are available.")

    if args.nproc == 1:
        main(rank=0)
    else:
        torch.multiprocessing.spawn(main, nprocs=args.nproc)