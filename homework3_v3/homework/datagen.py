def generate_dataset(
    output_json: str = "data/rft.json",
    oversample: int = 10,
    temperature: float = 0.6,
    batch_size: int = 4,
    max_questions: int = 0,
):
    import json
    from pathlib import Path

    from tqdm import tqdm

    from .cot import CoTModel
    from .data import Dataset, is_answer_valid

    model = CoTModel()
    dataset = Dataset("train")

    n = len(dataset) if max_questions <= 0 else min(max_questions, len(dataset))
    questions = [dataset[i][0] for i in range(n)]
    correct_answers = [dataset[i][1] for i in range(n)]

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume from existing file if present
    if output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        done_questions = {r[0] for r in results}
        print(f"Resuming: {len(results)} examples already saved.")
    else:
        results = []
        done_questions = set()

    successes = len(results)
    attempted = len(done_questions)

    print(f"Generating {oversample} completions per question (batch_size={batch_size}) on {n} questions...")

    for batch_start in tqdm(range(0, n, batch_size), desc="Datagen batches"):
        batch_qs = questions[batch_start : batch_start + batch_size]
        batch_ans = correct_answers[batch_start : batch_start + batch_size]

        # Skip questions already processed
        pending_idx = [i for i, q in enumerate(batch_qs) if q not in done_questions]
        if not pending_idx:
            continue

        pending_qs = [batch_qs[i] for i in pending_idx]
        pending_ans = [batch_ans[i] for i in pending_idx]
        prompts = [model.format_prompt(q) for q in pending_qs]

        all_completions = model.batched_generate(prompts, num_return_sequences=oversample, temperature=temperature)

        for question, correct_answer, completions in zip(pending_qs, pending_ans, all_completions):
            attempted += 1
            done_questions.add(question)
            for completion in completions:
                parsed = model.parse_answer(completion)
                if is_answer_valid(parsed, correct_answer):
                    results.append([question, correct_answer, completion])
                    successes += 1
                    break

        # Save incrementally after each batch
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    print(
        f"Done: {successes} / {attempted} ({successes / max(attempted, 1) * 100:.1f}% success rate) → {output_path}"
    )


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
