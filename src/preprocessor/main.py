from src.preprocessor.preprocessor import run_preprocessing

if __name__ == "__main__":
    run_preprocessing(
        input_path="raw/",      # <-- directory now supported
        output_dir="data/",
        d=64,
    )
