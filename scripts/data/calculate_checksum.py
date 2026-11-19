
import os
import hashlib

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
    "checksums.sha256"
)


def calculate_sha256(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as output:

        for root, dirs, files in os.walk(DATA_DIR):

            dirs.sort()
            files.sort()

            for filename in files:

                file_path = os.path.join(root, filename)

                # Skip KaggleHub completion marker
                if filename == "bundle.complete":
                    continue

                checksum = calculate_sha256(file_path)

                relative_path = os.path.relpath(
                    file_path,
                    PROJECT_ROOT
                )

                output.write(
                    f"{checksum}  {relative_path}\n"
                )

                print("Processed:", relative_path)

    print("\nChecksum completed!")
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
