from pathlib import Path

WORKSPACE = Path("./workspace").resolve()


def safe_path(path: str) -> Path:
    target = (WORKSPACE / path).resolve()

    if not str(target).startswith(str(WORKSPACE)):
        raise ValueError("Access outside workspace is not allowed")

    return target


def list_files(path: str = "."):
    target = safe_path(path)

    if not target.exists():
        return f"Path does not exist: {path}"

    return "\n".join(
        str(p.relative_to(WORKSPACE))
        for p in target.iterdir()
    )


def read_file(path: str):
    target = safe_path(path)

    if not target.is_file():
        return f"Not a file: {path}"

    return target.read_text()


def write_file(path: str, content: str):
    target = safe_path(path)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)

    return f"File written: {path}"


def create_directory(path: str):
    target = safe_path(path)

    target.mkdir(parents=True, exist_ok=True)

    return f"Directory created: {path}"

