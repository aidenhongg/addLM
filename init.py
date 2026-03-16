import random
from datasets import load_dataset

def generate_math_equations(size: int = 100000) -> list[tuple[int, int, str]]:
    equations = []
    for _ in range(size // 2):
        a = random.randint(0, 1000000)
        b = random.randint(0, 1000000)
        equations.append((a, b, "+"))
        equations.append((a, b, "-"))
    return equations


def download_datasets(max_math_stories: int = 500_000, max_tiny_stories: int = 500_000):
    math_stories = load_dataset("azminetoushikwasi/math-story-problems")
    tiny_stories = load_dataset("roneneldan/TinyStories")
    math_equations = generate_math_equations()

    # Cap rows per split
    for split in math_stories:
        n = len(math_stories[split])
        if n > max_math_stories:
            math_stories[split] = math_stories[split].select(range(max_math_stories))
    for split in tiny_stories:
        n = len(tiny_stories[split])
        if n > max_tiny_stories:
            tiny_stories[split] = tiny_stories[split].select(range(max_tiny_stories))

    return { "math_stories": math_stories, 
            "tiny_stories": tiny_stories, 
            "math_equations": math_equations }

if __name__ == "__main__":
    download_datasets()