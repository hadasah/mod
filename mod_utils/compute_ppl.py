from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import torch.nn.functional as F
import sys
from evaluate import load


def ppl(path, prompts):
    perplexity = load("perplexity", module_type="metric")
    return perplexity.compute(input_texts=prompts, model_id=path,device='gpu')

def log_probs_with_ppl(path, prompt):
    model = AutoModelForCausalLM.from_pretrained(path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)

    logits = outputs.logits
    arg_probs, _ = F.softmax(logits, dim=-1).max(-1)
    print("argmax probility:", arg_probs[0].cpu().detach().numpy())
    log_probs, tokens = F.log_softmax(logits, dim=-1).max(-1)
    print("argmax log probability:", log_probs[0].cpu().detach().numpy())
    sent = tokenizer.decode(tokens.squeeze().cpu().detach().numpy(), skip_special_tokens=False)
    print("argmax tokens:", sent)
    xentropy_loss = outputs[0]
    print("cross entropy loss:", xentropy_loss.item())
    ppl = torch.exp(xentropy_loss).item()
    print("ppl:", ppl)


if __name__ == "__main__":
    file = sys.argv[1]

    with open(file, 'r') as f:
        prompts = f.read().split("<|endoftext|>")
        prompts = [x for x in prompts if len(x) > 1]

    for model_id in ["opt-125m", "opt-350m", "opt-1.3b"]:
        print(20 * "=" + model_id + 20 * "=")
        model_path = os.path.join("facebook", model_id)
        print(ppl(model_path, prompts)['mean_perplexity'])
