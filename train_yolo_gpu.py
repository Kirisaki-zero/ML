from pathlib import Path
import os

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError(
        "ultralytics belum terpasang. Jalankan `pip install ultralytics` terlebih dahulu."
    ) from exc

try:
    import torch
except ImportError:
    torch = None

# Optimalkan PyTorch untuk mengurangi penggunaan memori
os.environ["TORCH_CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def print_gpu_memory():
    """Print GPU memory usage."""
    if torch is not None and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(0) / 1024**3    # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        print(f"GPU Memory - Allocated: {allocated:.2f}GB / Reserved: {reserved:.2f}GB / Total: {total:.2f}GB")


def main():
    root = Path(__file__).resolve().parent
    data_yaml = root / "Food_new.v2-allergen30.yolo26" / "data.yaml"
    save_dir = root / "runs" / "train"

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"File data.yaml tidak ditemukan: {data_yaml}\n"
            f"Pastikan folder dataset berada di lokasi yang benar."
        )

    if torch is None:
        raise RuntimeError(
            "PyTorch tidak ditemukan. Install PyTorch CUDA terlebih dahulu dan jalankan kembali."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA tidak tersedia di environment ini. Pastikan Anda menggunakan interpreter Anaconda dengan PyTorch CUDA."
        )

    device = "cuda:0"
    print("=== Konfigurasi Training ===")
    print(f"data:  {data_yaml}")
    print(f"device: {device}")
    print(f"save:   {save_dir}")
    print(f"torch: {torch.__version__}")
    print(f"cuda: {torch.version.cuda}")
    print(f"gpu:   {torch.cuda.get_device_name(0)}")
    print_gpu_memory()
    print()

    model = YOLO("yolo26n.pt")  # Menggunakan model YOLOv26 nano (ganti ke 'm' atau tipe lain jika GPU mumpuni)
    print("Memulai training dengan konfigurasi berikut:")
    print(f"Model: {model}")
    print(f"Device: {device}")
    print_gpu_memory()
    print("Memulai proses training...")

    model.train(
        data=str(data_yaml),
        epochs=40,
        imgsz=320,             # Lebih kecil = lebih hemat VRAM
        device=device,
        project=str(save_dir),
        name="my_yolo_train",
        exist_ok=True,
        # === GPU Only (minimal RAM) ===
        batch=8,               # Sangat kecil agar data langsung ke GPU VRAM
        workers=0,             # 0 = main thread only (hemat CPU/RAM)
        patience=20,           # Early stopping
        amp=True,              # Automatic Mixed Precision (hemat VRAM 30-50%)
        cache=False,           # Jangan cache ke RAM
        close_mosaic=10,       # Close mosaic augmentation 10 epoch terakhir (hemat memory)
        # === Training Config ===
        freeze=10,             # Freeze backbone
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # === Performa GPU ===
        half=True,             # FP16 precision (lebih cepat)
        verbose=True,
    )

    print_gpu_memory()
    print("Training selesai. Hasil tersimpan di:", save_dir / "my_yolo_train")


if __name__ == "__main__":
    main()
