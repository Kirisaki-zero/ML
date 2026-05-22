from pathlib import Path
import os
import gc

# Batasi penggunaan thread CPU agar hemat RAM
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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


def print_gpu_memory():
    """Print GPU memory usage."""
    if torch is not None and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024 ** 3  # GB
        reserved = torch.cuda.memory_reserved(0) / 1024 ** 3  # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3  # GB
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
    print("=== Konfigurasi Training Mode Akselerasi ===")
    print(f"data:  {data_yaml}")
    print(f"device: {device}")
    print(f"save:   {save_dir}")

    gc.collect()
    torch.cuda.empty_cache()  # Kosongkan cache GPU sebelum mulai

    # Memuat model YOLO26 versi NANO
    model = YOLO("yolo26n.pt")

    print("Memulai training dengan konfigurasi berikut:")
    print(f"Model: {model}")
    print(f"Device: {device}")
    print_gpu_memory()
    print("Memulai proses training...")

    model.train(
        data=str(data_yaml),
        epochs=100,  # Dibatasi ketat di 100 putaran
        imgsz=416,  # Resolusi optimal untuk versi Nano

        # === AKSELERASI & GPU MAXIMIZER ===
        batch=8,  # Memaksimalkan VRAM 6GB agar cepat selesai
        optimizer='AdamW',  # Mesin belajar cepat
        lr0=0.002,  # Kecepatan belajar awal

        device=device,
        project=str(save_dir),
        name="my_yolo_train_100ep",
        exist_ok=True,

        # === PENGAMANAN RAM WINDOWS ===
        workers=0,  # RAM dijamin aman, tidak akan loncat ke 90%
        cache=False,

        # === PENGURANGAN SOAL SULIT (AUGMENTASI RINGAN) ===
        mixup=0.0,  # Dimatikan agar tidak bingung
        degrees=5.0,  # Rotasi ringan maksimal 5 derajat
        translate=0.1,  # Pergeseran ringan 10%
        scale=0.5,
        shear=0.0,  # Distorsi kemiringan dimatikan
        fliplr=0.5,  # Mirror horizontal tetap 50%

        # === FINISHING (MEMANTAPKAN KEYAKINAN) ===
        close_mosaic=20,  # 20 putaran terakhir model melihat gambar murni tanpa editan

        # === Save & Validation Config ===
        save=True,
        patience=50,
        amp=True,  # Wajib True agar fitur Tensor Core RTX 3050 menyala
        half=True,
        verbose=True,
    )

    print_gpu_memory()
    print("Training selesai. Hasil tersimpan di:", save_dir / "my_yolo_train_100ep")


if __name__ == "__main__":
    main()