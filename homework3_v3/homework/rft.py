from .base_llm import BaseLLM
from .sft import TokenizedDataset, tokenize, test_model


def load() -> BaseLLM:
    from pathlib import Path
    from peft import PeftModel

    model_path = Path(__file__).parent / "rft_model"

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    def _format_prompt(question: str) -> str:
        return question + "\n"

    llm.format_prompt = _format_prompt
    return llm


def format_rft_example(question: str, correct_answer, completion: str) -> dict:
    # completion already contains full reasoning + <answer>X</answer>
    return {"question": question, "answer": completion}


def train_model(output_dir: str = "rft_output", **kwargs):
    import json
    from pathlib import Path
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    rft_data_path = Path(__file__).parent.parent / "data" / "rft.json"
    if not rft_data_path.exists():
        raise FileNotFoundError(f"Run python -m homework.datagen first.")

    with open(rft_data_path) as f:
        rft_data = json.load(f)

    if len(rft_data) == 0:
        raise ValueError("rft.json is empty! Run python -m homework.datagen first.")

    print(f"Loaded {len(rft_data)} RFT examples. Sample: {rft_data[0]}")

    llm = BaseLLM()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    llm.model = get_peft_model(llm.model, lora_config)
    llm.model.enable_input_require_grads()
    llm.model.print_trainable_parameters()

    train_data = TokenizedDataset(
        llm.tokenizer,
        rft_data,
        format_rft_example,
        max_length=256,  # longer for reasoning chains
    )

    # Verify tokenization
    sample = train_data[0]
    ids = sample["input_ids"]
    labels = sample["labels"]
    visible = [ids[i] for i in range(len(labels)) if labels[i] != -100]
    print("=== RFT Tokenization check ===")
    print("Answer tokens trained on:", llm.tokenizer.decode(visible))
    print("==============================")

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="epoch",
        logging_steps=10,
        remove_unused_columns=False,
        fp16=False,
        bf16=False,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=train_data,
    )
    trainer.train()

    save_path = Path(__file__).parent / "rft_model"
    trainer.save_model(str(save_path))
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    from fire import Fire
    Fire({"train": train_model, "test": test_model, "load": load})