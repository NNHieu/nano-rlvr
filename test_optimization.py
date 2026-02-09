import sys
from unittest.mock import MagicMock
import torch
import torch.nn.functional as F

# Mock modules
sys.modules["deepspeed"] = MagicMock()
sys.modules["wandb"] = MagicMock()
sys.modules["vllm"] = MagicMock()
sys.modules["datasets"] = MagicMock()
sys.modules["transformers"] = MagicMock()

# Mock specific classes imported
sys.modules["deepspeed"].DeepSpeedEngine = MagicMock
sys.modules["transformers"].PreTrainedModel = MagicMock
sys.modules["transformers"].AutoTokenizer = MagicMock

from utils import compute_token_log_probs, log_softmax_and_gather

def compute_token_log_probs_old(
    model,
    inputs,
    temperature,
):
    # Mocking model output
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        return_dict=True,
        use_cache=False,
    )

    logits = outputs.logits / temperature
    shift_logits = logits[..., :-1, :]
    shift_labels = inputs["labels"][..., 1:].clone() # Clone because original modified it in place!
    shift_labels_mask = inputs["labels_mask"][..., 1:]

    # Create mask for valid labels
    shift_labels[~(shift_labels_mask.bool())] = 0

    # Calculate log probabilities
    log_probs = log_softmax_and_gather(shift_logits, shift_labels)
    log_probs = log_probs * shift_labels_mask

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

def test():
    torch.manual_seed(42)
    vocab_size = 50
    batch_size = 2
    seq_len = 10
    
    inputs = {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "labels_mask": torch.ones(batch_size, seq_len),
    }
    
    # Add some -100 in labels and 0 in mask
    inputs["labels"][0, 5] = -100
    inputs["labels_mask"][0, 5] = 0
    # Also mask last token
    inputs["labels"][1, 8] = -100
    inputs["labels_mask"][1, 8] = 0

    # Create fixed logits
    logits = torch.randn(batch_size, seq_len, vocab_size)
    model = MockModel(logits)
    temperature = 0.8
    
    # Run old
    log_probs_old = compute_token_log_probs_old(model, inputs, temperature)
    
    # Run new
    log_probs_new = compute_token_log_probs(model, inputs, temperature)
    
    print("Old shape:", log_probs_old.shape)
    print("New shape:", log_probs_new.shape)
    
    # Check close
    diff = (log_probs_old - log_probs_new).abs().max()
    print("Max difference:", diff.item())
    
    if diff < 1e-4:
        print("PASS")
    else:
        print("FAIL")
        print("Old:", log_probs_old)
        print("New:", log_probs_new)

if __name__ == "__main__":
    test()
