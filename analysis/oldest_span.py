from pathlib import Path


def main() -> None:
    # Resolve the analysis directory relative to the repo root so this script
    # works regardless of where the project is checked out.
    repo_root = Path(__file__).resolve().parents[1]
    root = (
        repo_root
        / "artifacts" / "output" / "travel" / "analysis" / "query_load"
        / "gpt5mini_react" / "gpt5mini_test" / "two_stage_gpt5mini_test"
    )
    files = [p for p in root.iterdir() if p.is_file()]
    if not files:
        print(f"No files found under: {root}")
        return

    files_sorted = sorted(files, key=lambda p: p.stat().st_mtime)
    oldest = files_sorted[:101]
    if len(oldest) < 2:
        print(f"Not enough files to compute span: {len(oldest)}")
        return

    t0 = oldest[0].stat().st_mtime
    t1 = oldest[-1].stat().st_mtime
    print("count:", len(oldest))
    print("first:", oldest[0].name, "mtime:", t0)
    print("last:", oldest[-1].name, "mtime:", t1)
    print("span_s:", t1 - t0)


if __name__ == "__main__":
    main()
