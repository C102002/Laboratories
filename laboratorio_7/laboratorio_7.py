import os, math, sys
import torch
import torchaudio
from huggingsound import SpeechRecognitionModel

def split_mp3_torch(mp3_path: str,
                    out_dir: str,
                    segment_length_s: int = 30) -> list[str]:
    """
    Carga un MP3 con torchaudio, lo parte en WAVs de segment_length_s,
    los guarda con torchaudio.save y devuelve la lista de rutas.
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) Carga con torchaudio
    try:
        waveform, sr = torchaudio.load(mp3_path)  # (canales, muestras)
    except Exception as e:
        print(f"❌ Error cargando '{mp3_path}' con torchaudio: {e}")
        sys.exit(1)

    total_samples   = waveform.shape[1]
    samples_per_seg = segment_length_s * sr
    n_parts         = math.ceil(total_samples / samples_per_seg)

    wav_paths = []
    for i in range(n_parts):
        start = int(i * samples_per_seg)
        end   = int(min((i + 1) * samples_per_seg, total_samples))
        chunk = waveform[:, start:end]

        out_wav = os.path.join(out_dir, f"segment_{i:03d}.wav")
        try:
            torchaudio.save(out_wav, chunk, sr, format="wav")
            wav_paths.append(out_wav)
        except Exception as e:
            print(f"⚠️ Error guardando segmento {i}: {e}")

    return wav_paths

if __name__ == "__main__":
    mp3_file     = "Clase IA Procesamiento del lenguaje natural.mp3"
    segments_dir = "segments"

    print("🔊 Extrayendo y segmentando audio con torchaudio…")
    wavs = split_mp3_torch(mp3_file, segments_dir, segment_length_s=30)
    if not wavs:
        print("‼️ No se generaron WAVs. Abortando.")
        sys.exit(1)

    print("⚙️  Cargando modelo…")
    model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-spanish")
    
    # ——— Detecta y usa GPU si existe ———
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"→ Usando dispositivo: {device}")

    # Ahora al llamar a transcribe() el modelo ya está en GPU
    print(f"📝 Transcribiendo {len(wavs)} segmentos…")
    results = model.transcribe(
        wavs,
        batch_size=4,
    )

    print(f"📝 Transcribiendo {len(wavs)} segmentos…")
    try:
        results = model.transcribe(wavs, batch_size=4)
    except Exception as e:
        print(f"❌ Error en la transcripción: {e}")
        sys.exit(1)

    # Concatenar en orden temporal
    full_text = ""
    if isinstance(results, dict):
        for path in sorted(results):
            full_text += results[path] + " "
    else:
        full_text = " ".join(results)

    print("\n--- Transcripción Completa ---")
    print(full_text[:500], "…")  # opcional: ventana previa

    # 6) Guardar en archivo de texto
    out_path = "transcripcion_completa.txt"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"\n✅ Transcripción guardada en '{out_path}'")
    except Exception as e:
        print(f"\n❌ Error al guardar archivo: {e}")

