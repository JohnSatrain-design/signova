from __future__ import annotations

from signova.config import ensure_project_dirs
from signova.training.dataset import DatasetManager


def main() -> None:
    ensure_project_dirs()
    manager = DatasetManager()
    counts = manager.sample_counts()
    width = max([len(label) for label in counts] + [5])
    print("SIGNOVA dataset summary")
    print("-" * (width + 12))
    for label, count in counts.items():
        print(f"{label:<{width}}  {count:>6}")


if __name__ == "__main__":
    main()
