from datasets import load_dataset


def download_datasets(max_math_stories: int = 500_000, max_stories: int = 500_000, max_analogies: int = 500_000):
    math_stories = load_dataset("azminetoushikwasi/math-story-problems")
    stories = {"train": load_dataset(
        "HuggingFaceTB/cosmopedia", "stories",
        split=f"train[:{max_stories}]", trust_remote_code=True,
    )}
    analogies = load_dataset("saturnMars/hyperprobe-dataset-analogy")

    # Cap rows per split
    for split in math_stories:
        n = len(math_stories[split])
        if n > max_math_stories:
            math_stories[split] = math_stories[split].select(range(max_math_stories))
    for split in analogies:
        n = len(analogies[split])
        if n > max_analogies:
            analogies[split] = analogies[split].select(range(max_analogies))

    return {
        "math_stories": math_stories,
        "stories": stories,
        "analogies": analogies,
    }


if __name__ == "__main__":
    download_datasets()