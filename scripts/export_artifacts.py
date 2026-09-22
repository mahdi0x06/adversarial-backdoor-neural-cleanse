from pathlib import Path
import shutil


OUTPUT_ROOT = Path(
    "outputs"
)

ARCHIVE_NAME = (
    "adversarial_backdoor_neural_cleanse_artifacts"
)


def main():

    archive_path = shutil.make_archive(
        ARCHIVE_NAME,
        "zip",
        root_dir=OUTPUT_ROOT,
    )

    print(
        "Created:",
        archive_path,
    )


if __name__ == "__main__":
    main()