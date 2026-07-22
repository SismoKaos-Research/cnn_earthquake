from pathlib import Path
from typing import Iterable, List

from src.preprocessor.ram_transformation import mseed_3ch_to_ram_rgb


def iter_mseed_files(input_path: str | Path) -> List[Path]:
    """
    Accept either:
      - a single .mseed file
      - a directory containing .mseed files (e.g., raw/)
    Returns sorted list of files to process.
    """
    p = Path(input_path)

    if not p.exists():
        raise FileNotFoundError(f"Input path not found: {p}")

    if p.is_file():
        if p.suffix.lower() not in {".mseed", ".msd"}:
            raise ValueError(f"Expected MiniSEED file, got: {p}")
        return [p]

    files = sorted(
        [f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in {".mseed", ".msd"}]
    )
    if not files:
        raise ValueError(f"No MiniSEED files found under directory: {p}")
    return files


def run_preprocessing(input_path: str, output_dir: str, d: int = 64):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mseed_files = iter_mseed_files(input_path)
    print(f"Found {len(mseed_files)} MiniSEED file(s) in: {input_path}")

    for fp in mseed_files:
        try:
            print(f"Processing: {fp}")
            out_png = out / f"{fp.stem}_ram.png"
            mseed_3ch_to_ram_rgb(str(fp.absolute()), str(out_png), d=d)
        except Exception as e:
            print(f"[WARN] Failed {fp}: {e}")
