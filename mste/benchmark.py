"""Official KVC-onGoing benchmark files: test comparison pairs in, similarity scores out."""
import os


def load_comparisons(scenario: str, logger) -> list[tuple[str, str]]:
    """Load comparison pairs from file."""
    comps_file = f"data/raw/{scenario}/{scenario}_comparisons.txt"

    with open(comps_file, "r") as f:
        comparisons = eval(f.readline())

    logger.info(f"Loaded {len(comparisons)} comparison pairs")
    return comparisons

def save_predictions(similarities: list[float], scenario: str, run_dir: str, logger) -> None:
    """Save similarity scores to file."""
    os.makedirs(run_dir, exist_ok=True)
    output_file = os.path.join(run_dir, f'{scenario}_predictions.txt')
    with open(output_file, "w") as f:
        f.write(str(similarities))
    logger.info(f"Predictions saved to {output_file}")
