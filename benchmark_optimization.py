import sys
import time
import torch
import torch.nn.functional as F
from unittest.mock import MagicMock

# Mock modules
sys.modules["deepspeed"] = MagicMock()
sys.modules["wandb"] = MagicMock()
sys.modules["vllm"] = MagicMock()
sys.modules["datasets"] = MagicMock()
sys.modules["transformers"] = MagicMock()

# Mock specific classes
sys.modules["deepspeed"].DeepSpeedEngine = MagicMock
sys.modules["transformers"].PreTrainedModel = MagicMock
sys.modules["transformers"].AutoTokenizer = MagicMock

# Import AFTER mocking
from utils import compute_token_log_probs, log_softmax_and_gather

# 1. Original (reconstructed)
def compute_token_log_probs_original(model, inputs, temperature):
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        return_dict=True,
        use_cache=False,
    )

    logits = outputs.logits / temperature
    shift_logits = logits[..., :-1, :]
    shift_labels = inputs["labels"][..., 1:].clone() 
    shift_labels_mask = inputs["labels_mask"][..., 1:]

    shift_labels[~(shift_labels_mask.bool())] = 0

    log_probs = log_softmax_and_gather(shift_logits, shift_labels)
    log_probs = log_probs * shift_labels_mask

    return log_probs

# 2. Current Implementation in file (Copy based Cross Entropy)
# Imported as compute_token_log_probs

# 3. Candidate: Permuted Cross Entropy (Avoids contiguous copy)
def compute_token_log_probs_permuted(model, inputs, temperature):
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        return_dict=True,
        use_cache=False,
    )

    shift_logits = outputs.logits[..., :-1, :]
    shift_labels = inputs["labels"][..., 1:]

    # For benchmark Temp=1.0, this is 0-copy.
    # If Temp != 1.0, we pay copy cost unless we fuse.
    if temperature != 1.0:
        shift_logits = shift_logits / temperature

    # Permute to (Batch, Vocab, SeqLen) for cross_entropy
    shift_logits = shift_logits.permute(0, 2, 1)
    
    log_probs = -F.cross_entropy(
        shift_logits,
        shift_labels,
        reduction="none",
        ignore_index=-100,
    )

    return log_probs

# 4. Candidate: Fused Torch Compile
def fused_log_softmax_gather(logits, index, temperature):
    return (logits / temperature).log_softmax(dim=-1).gather(dim=-1, index=index.unsqueeze(-1)).squeeze(-1)

compiled_fused_op = torch.compile(fused_log_softmax_gather, dynamic=True)

def compute_token_log_probs_fused(model, inputs, temperature):
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        return_dict=True,
        use_cache=False,
    )
    
    shift_logits = outputs.logits[..., :-1, :] # View
    shift_labels = inputs["labels"][..., 1:]   # View
    shift_labels_mask = inputs["labels_mask"][..., 1:]
    
    # We need to handle masking. The original did strict masking.
    # Fused op will compute for whatever index is there.
    # To match behavior, we can rely on caller handling mask or do it here.
    # Original logic: shift_labels[~mask] = 0
    # To avoid modify-in-place, we can pass mask or assume we just mask output.
    # But invalid labels (-100) are problem for gather if not handled.
    # Let's clone labels for safety in this candidate, or use a safe fill
    
    # Ideally we'd just use -100 handling from cross_entropy? 
    # But for gather, index must be valid.
    
    # Cheap clone of integer labels is fine (small compared to logits)
    shift_labels_safe = shift_labels.clone()
    shift_labels_safe[shift_labels == -100] = 0
    
    log_probs = compiled_fused_op(shift_logits, shift_labels_safe, temperature)
    
    # Mask out results where label was -100
    log_probs = log_probs * (shift_labels != -100)
    
    return log_probs


class MockModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits_tensor = logits

    def forward(self, input_ids, **kwargs):
        class Output:
            pass
        out = Output()
        out.logits = self.logits_tensor
        return out

def benchmark():
    print("Setting up benchmark...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    batch_size = 2
    seq_len = 1024
    vocab_size = 32000
    
    print(f"Configuration: Batch={batch_size}, SeqLen={seq_len}, Vocab={vocab_size}")
    
    inputs = {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len), device=device),
        "attention_mask": torch.ones(batch_size, seq_len, device=device),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_len), device=device),
        "labels_mask": torch.randint(0, 2, (batch_size, seq_len), device=device),
    }
    # Ensure -100 is present to test handling
    inputs["labels"][0, :10] = -100
    
    logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
    model = MockModel(logits)
    temperature = 0.8

    def measure(func, name):
        # Reset memory
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
        # Warmup
        try:
            for _ in range(5):
                func(model, inputs, temperature)
        except Exception as e:
            print(f"{name} Failed: {e}")
            return 0
        
        start_mem = 0
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        start_time = time.time()
        iters = 20
        for _ in range(iters):
            func(model, inputs, temperature)
        
        if device == "cuda":
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated()
            mem_used = peak_mem / 1024**2 # Total peak recorded
        else:
            mem_used = 0 
            
        end_time = time.time()
        avg_time = (end_time - start_time) / iters
        
        print(f"{name}:")
        print(f"  Avg Time: {avg_time*1000:.2f} ms")
        if device == "cuda":
            print(f"  Peak Memory: {mem_used:.2f} MB")
        
        return avg_time

    print("\nRunning benchmarks...")
    measure(compute_token_log_probs_original, "Original")
    measure(compute_token_log_probs, "Current (In-place/Contiguous)")
    measure(compute_token_log_probs_permuted, "Permuted CrossEntropy")
    measure(compute_token_log_probs_fused, "Fused Compile")

if __name__ == "__main__":
    benchmark()
