import argparse
import torch
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from model.peft import MoeLoraConfig, get_peft_model
from model.peft.tuners.gating import GATING_TO_MODEL_MAPPING


def parse_args():
    parser = argparse.ArgumentParser(description="Demo for applying MoeLoRA to a model")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to pretrained model. If omitted, a tiny random Llama model is used.",
    )
    return parser.parse_args()


def load_base_model(model_path: str):
    if model_path is not None:
        cfg = AutoConfig.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_config(cfg)
    else:
        cfg = LlamaConfig(
            vocab_size=100,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
        )
        model = LlamaForCausalLM(cfg)
    return model, cfg


def main():
    args = parse_args()

    base_model, model_cfg = load_base_model(args.model_path)

    # Configure MoeLoRA
    moelora_cfg = MoeLoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=4,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=None,
        num_moe=2,
        gating="Dense",
    )

    model = get_peft_model(base_model, moelora_cfg)
    model.print_trainable_parameters()

    # Simple gating network
    gate_model = GATING_TO_MODEL_MAPPING[moelora_cfg.gating](
        dim=getattr(model_cfg, "hidden_size", 64), num_moe=moelora_cfg.num_moe
    )

    batch_size, seq_len = 2, 5
    input_ids = torch.randint(0, model_cfg.vocab_size, (batch_size, seq_len))
    user_embeds = torch.randn(batch_size, gate_model.dim)
    gate_weights = gate_model(user_embeds).unsqueeze(1)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, user_embeds=user_embeds, gate_weights=gate_weights)
    print("Output logits shape:", outputs.logits.shape)


if __name__ == "__main__":
    main()
