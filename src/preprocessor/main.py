from src.preprocessor.preprocessor import run_preprocessing

if __name__ == "__main__":
    run_preprocessing(
        input_path="raw/",      
        output_dir="data/",
        d=64,
        target_fs=100.0,
        window_seconds=200.0,
        overlap=0.75
    )
