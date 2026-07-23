from pathlib import Path
from typing import Iterable, List

from src.preprocessor.ram_transformation import mseed_to_ram_rgb


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


def run_preprocessing(
    input_path: str, 
    output_dir: str, 
    d: int = 64, 
    target_fs: float = 100.0,
    window_seconds: float = 60.0,
    overlap: float = 0.5
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mseed_files = iter_mseed_files(input_path)
    print(f"Found {len(mseed_files)} MiniSEED file(s) in: {input_path}")

    for fp in mseed_files:
        try:
            print(f"Processing: {fp}")
            
            out_png = out / f"{fp.stem}_ram.png"
            
            mseed_to_ram_rgb(
                mseed_path=str(fp.absolute()), 
                out_png=str(out_png), 
                d=d, 
                target_fs=target_fs,
                window_seconds=window_seconds,
                overlap=overlap
            )
        except Exception as e:
            print(f"[WARN] Failed {fp}: {e}")


if __name__ == "__main__":
    run_preprocessing(
        input_path="raw/",      
        output_dir="data/",
        d=64,
        target_fs=100.0,
        window_seconds=200.0,
        overlap=0.75
    )
