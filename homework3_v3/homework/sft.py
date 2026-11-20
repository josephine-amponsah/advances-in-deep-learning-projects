from .base_llm import BaseLLM
from .data import Dataset, benchmark


def load() -> BaseLLM:
    from pathlib import Path
    from peft import PeftModel

    model_path = Path(__file__).parent / "sft_model"

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    def _format_prompt(question: str) -> str:
        return question + "\n"

    llm.format_prompt = _format_prompt
    return llm


def tokenize(tokenizer, question: str, answer: str, max_length: int = 128):
    # Newline separator prevents BPE merge between last question token and first answer token
    full_text = f"{question}\n{answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=max_length)
    input_ids = full["input_ids"]
    question_len = len(tokenizer(question + "\n")["input_ids"])

    labels = [-100] * question_len + input_ids[question_len:]
    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_example(prompt: str, answer) -> dict:
    ans = float(answer)
    rounded = round(ans, 3)
    answer_str = str(int(rounded)) if rounded == int(rounded) else str(rounded)
    return {"question": prompt, "answer": f"<answer>{answer_str}</answer>"}


class TokenizedDataset:
    def __init__(self, tokenizer, data, format_fn, max_length: int = 128):
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formatted = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, max_length=self.max_length, **formatted)


def train_model(output_dir: str = "sft_output", **kwargs):
    from pathlib import Path
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    llm = BaseLLM()

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    llm.model = get_peft_model(llm.model, lora_config)
    llm.model.enable_input_require_grads()
    llm.model.print_trainable_parameters()

    train_data = TokenizedDataset(
        llm.tokenizer, Dataset("train"), format_example, max_length=128
    )

    # Verify tokenization looks correct on one example
    sample = train_data[0]
    ids = sample["input_ids"]
    labels = sample["labels"]
    visible = [ids[i] for i in range(len(labels)) if labels[i] != -100]
    print("=== Tokenization check ===")
    print("Answer tokens the model trains on:", llm.tokenizer.decode(visible))
    print("==========================")

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
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

    save_path = Path(__file__).parent / "sft_model"
    trainer.save_model(str(save_path))
    print(f"Model saved to {save_path}")


def test_model(ckpt_path: str = ""):
    testset = Dataset("valid")
    llm = load()
    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire
    Fire({"train": train_model, "test": test_model, "load": load})