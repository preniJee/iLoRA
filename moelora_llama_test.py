import torch
from transformers import LlamaConfig, LlamaForCausalLM

from model.peft import MoeLoraConfig, get_peft_model
from model.peft.tuners.gating import GATING_TO_MODEL_MAPPING


def main():
    # Create a tiny Llama model from scratch
    llama_cfg = LlamaConfig(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
    )
    base_model = LlamaForCausalLM(llama_cfg)

    # Configure MoeLoRA
    moelora_cfg = MoeLoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=4,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        num_moe=2,
        gating="Dense",
    )

    model = get_peft_model(base_model, moelora_cfg)
    model.print_trainable_parameters()

    # Simple gating network
    gate_model = GATING_TO_MODEL_MAPPING["Dense"](dim=64, num_moe=moelora_cfg.num_moe)

    batch_size, seq_len = 2, 5
    input_ids = torch.randint(0, llama_cfg.vocab_size, (batch_size, seq_len))
    user_embeds = torch.randn(batch_size, 64)
    gate_weights = gate_model(user_embeds).unsqueeze(1)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, user_embeds=user_embeds, gate_weights=gate_weights)
    print("Output logits shape:", outputs.logits.shape)


if __name__ == "__main__":
    main()
