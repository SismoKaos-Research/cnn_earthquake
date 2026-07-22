import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    SCRIPT_DIR: Path = Path('.').resolve()
    STATION: str = 'ELZG'
    EARTHQUAKE_NAME: str = '24012020_M6.8_Sivrice__Elazig_'
    RAW_FILE_NAME: str = 'TU_ELZG_24012020_000000_25012020_000000_HH.mseed'

    # Preprocessor parameters

    PREPROCESS_WINDOW_SEC: float = 3600.0
    FREQMIN: float = 0.1
    FREQMAX: float = 2.0
    GAP_THRESHOLD: float = 2.0
    SAVE_REPORT: bool = True
    SAVE_CSV:bool = True

    # Feature extraction
    Fs: float = 5.0
    WIN_SEC: int = 200
    STEP_SEC: int = 50
    PREV_SEC: int = 150
    
    STA_SEC: float = 0.5
    LTA_SEC: int = 60
    
    N_JOBS: int = -1
    WARMUP_COUNT: int = 3
    PRINT_LINE: bool = False

    # Dynamic parameters
    MSEED_INPUT_FILE: Path = field(init=False)
    DATA_ROOT: Path = field(init=False)
    OUTPUT_ROOT: Path = field(init=False)
    
    OVERLAP: float = field(init=False)
    WinSize: int = field(init=False)
    StepSize: int = field(init=False)
    PREV_LEN: int = field(init=False)
    NSTA: int = field(init=False)
    NLTA: int = field(init=False)

    def __post_init__(self):
        # dis how u do it if its frozen dataclass
        object.__setattr__(self, 'MSEED_INPUT_FILE', self.SCRIPT_DIR / "raw" / self.STATION / self.EARTHQUAKE_NAME / self.RAW_FILE_NAME)
        object.__setattr__(self, 'DATA_ROOT', self.SCRIPT_DIR / "data" / self.EARTHQUAKE_NAME)
        object.__setattr__(self, 'OUTPUT_ROOT', self.SCRIPT_DIR / 'results' / self.STATION)
        
        self.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        self.DATA_ROOT.mkdir(parents=True, exist_ok=True)
        
        object.__setattr__(self, 'OVERLAP', self.STEP_SEC / self.WIN_SEC)
        object.__setattr__(self, 'WinSize', int(self.WIN_SEC * self.Fs))
        object.__setattr__(self, 'StepSize', int(self.STEP_SEC * self.Fs))
        object.__setattr__(self, 'PREV_LEN', int(self.PREV_SEC * self.Fs))
        object.__setattr__(self, 'NSTA', int(self.STA_SEC * self.Fs))
        object.__setattr__(self, 'NLTA', int(self.LTA_SEC * self.Fs))

    @classmethod
    def from_json(cls, json_path: str | Path):
        """JSON dosyasından ayarları okur ve Settings objesi oluşturur."""
        path = Path(json_path)
        if not path.exists():
            print(f"[UYARI] {path} bulunamadı, varsayılan ayarlar kullanılıyor.")
            return cls()
            
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Sadece dataclass'ın init fonksiyonunda kabul ettiği anahtarları al
        valid_keys = {f.name for f in cls.__dataclass_fields__.values() if f.init}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        
        return cls(**filtered_data)

    def to_json(self, json_path: str | Path):
        """Mevcut ayarları JSON dosyasına kaydeder."""
        data = {}
        for k, v in asdict(self).items():
            # Path objeleri JSON'a doğrudan yazılamaz, string'e çeviriyoruz
            if isinstance(v, Path):
                data[k] = str(v)
            else:
                data[k] = v
                
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def config_info(self):
        print(f"NSTA = {self.NSTA} sample ({self.STA_SEC} sn)")
        print(f"NLTA = {self.NLTA} sample ({self.LTA_SEC} sn)")
        print(f"WinSize = {self.WinSize} sample ({self.WIN_SEC} sn)")
        print(f"DATA_ROOT  : {self.DATA_ROOT}")
        print(f"OUTPUT_ROOT: {self.OUTPUT_ROOT}")

# Projenin geri kalanında import edilecek instance
settings = Settings()
