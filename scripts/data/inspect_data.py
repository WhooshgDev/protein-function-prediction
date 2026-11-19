
import os
import csv

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "cafa5"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "metadata"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "file_inventory.csv"
)


def get_file_size(file_path):
    size_bytes = os.path.getsize(file_path)

    size_mb = size_bytes / (1024 * 1024)

    return size_bytes, size_mb


def get_file_extension(filename):
    _, extension = os.path.splitext(filename)

    return extension.lower()


def inspect_files():
    if not os.path.exists(DATA_DIR):
        print("Data directory not found!")
        print("Expected path:", DATA_DIR)
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    inventory = []

    for root, dirs, files in os.walk(DATA_DIR):

        dirs.sort()
        files.sort()

        for filename in files:

            file_path = os.path.join(root, filename)

            relative_path = os.path.relpath(
                file_path,
                PROJECT_ROOT
            )

            size_bytes, size_mb = get_file_size(file_path)

            extension = get_file_extension(filename)

            file_info = {
                "file_name": filename,
                "relative_path": relative_path,
                "extension": extension,
                "size_bytes": size_bytes,
                "size_mb": round(size_mb, 3)
            }

            inventory.append(file_info)

            print(
                f"{filename} | "
                f"{extension} | "
                f"{size_mb:.3f} MB"
            )

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        fieldnames = [
            "file_name",
            "relative_path",
            "extension",
            "size_bytes",
            "size_mb"
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(inventory)

    print("\nTotal files:", len(inventory))
    print("Inventory saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    inspect_files()
