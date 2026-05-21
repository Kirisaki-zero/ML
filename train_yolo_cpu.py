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


def print_cpu_info():
    """Print CPU & memory info."""
    import platform
    print(f"CPU: {platform.processor()}")
    print(f"CPU Cores (logical): {os.cpu_count()}")
    if torch is not None:
        print(f"torch: {torch.__version__}")
        print(f"OMP_NUM_THREADS: {torch.get_num_threads()} thread(s)")


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
            "PyTorch tidak ditemukan. Install PyTorch terlebih dahulu dan jalankan kembali."
        )

    # Paksa training di CPU — tidak perlu CUDA
    device = "cpu"

    # Optimalkan PyTorch untuk CPU: gunakan semua core yang tersedia
    num_threads = os.cpu_count() or 4
    torch.set_num_threads(num_threads)

    print("=== Konfigurasi Training ===")
    print(f"data:   {data_yaml}")
    print(f"device: {device}")
    print(f"save:   {save_dir}")
    print_cpu_info()
    print()

    model = YOLO("yolo26n.pt")  # Menggunakan model YOLOv26 nano
    print("Memulai training dengan konfigurasi berikut:")
    print(f"Model: {model}")
    print(f"Device: {device}")
    print("Memulai proses training...")

    model.train(
        data=str(data_yaml),
        epochs=40,
        imgsz=320,             # Lebih kecil = lebih hemat RAM & lebih cepat di CPU
        device=device,
        project=str(save_dir),
        name="my_yolo_train",
        exist_ok=True,
        # === CPU-Friendly Config ===
        batch=4,               # Batch kecil agar tidak memakan terlalu banyak RAM
        workers=0,             # 0 = main thread only (hindari overhead multiprocessing di CPU)
        patience=20,           # Early stopping
        amp=False,             # Matikan AMP — FP16 tidak didukung di CPU
        cache=False,           # Jangan cache ke RAM agar hemat memori
        close_mosaic=10,       # Close mosaic augmentation 10 epoch terakhir (hemat memory)
        # === Training Config (sama dengan GPU) ===
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
        # === CPU: half=False karena FP16 tidak didukung ===
        half=False,            # Harus False di CPU
        verbose=True,
    )

    print("Training selesai. Hasil tersimpan di:", save_dir / "my_yolo_train")


if __name__ == "__main__":
    main()
