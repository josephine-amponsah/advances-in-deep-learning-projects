from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a unit conversion assistant. "
                    "State the conversion factor, multiply, then give the final numeric answer inside <answer></answer>. "
                    "Be concise."
                ),
            },
            {
                "role": "user",
                "content": "How many grams are there per 6 kg?",
            },
            {
                "role": "assistant",
                "content": "1 kg = 1000 g. 6 * 1000 = <answer>6000</answer>",
            },
            {
                "role": "user",
                "content": "What is the conversion of 3 feet to inches?",
            },
            {
                "role": "assistant",
                "content": "1 ft = 12 in. 3 * 12 = <answer>36</answer>",
            },
            {
                "role": "user",
                "content": question,
            },
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
