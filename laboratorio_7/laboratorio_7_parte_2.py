import os
import sys
import spacy
from spellchecker import SpellChecker

# Carga el modelo de spaCy una sola vez
nlp = spacy.load("es_core_news_sm")

spell = SpellChecker(language="es")

def correct_text(text: str) -> str:
    """
    Corrige ortografía word-by-word y va imprimiendo
    los cambios que realiza.
    """
    words       = text.split()
    corrected   = []
    changes     = 0

    print("🔧 Iniciando corrección ortográfica…")
    for idx, w in enumerate(words, start=1):
        cand = spell.correction(w) or w
        # Si cambió algo, lo mostramos
        if cand.lower() != w.lower():
            print(f"  [{idx:04d}] {w} → {cand}")
            changes += 1
        corrected.append(cand)

    print(f"✅ Corrección finalizada: {changes} palabra(s) modificada(s)\n")
    return " ".join(corrected)

def preprocess_text(text: str) -> list[str]:
    """Tokeniza, elimina stop-words/puntuación/espacios y lematiza."""
    doc = nlp(text)
    return [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]

# Funcion para imprimirlo mas estilizado

def pretty_print(clean_txt: str, tokens: list[str],
                 char_preview: int = 500,
                 token_preview: int = 10):
    # Cabecera
    print("🎬" + "─" * 50 + "🎬")
    print("✨ Transcripción limpia ✨\n")
    # Texto limpio (previa)
    preview = clean_txt[:char_preview]
    ellipsis = "…" if len(clean_txt) > char_preview else ""
    print(f"📄 {preview}{ellipsis}\n")
    
    # Tokens
    print("🗝️ Algunos tokens 🗝️\n")
    sample = tokens[:token_preview]
    for t in sample:
        print(f"   • {t}")
    if len(tokens) > token_preview:
        print("   • …")
    
    # Pie
    print("\n🎬" + "─" * 50 + "🎬")


def save_transcription_and_tokens(
    raw_text: str,
    tokens: list[str],
    out_dir: str = "data",
    base_name: str = "transcripcion"
):
    """
    Guarda dos archivos en `out_dir`:
      1) {base_name}_limpia.txt  ← texto limpio (tokens unidos con espacio)
      2) {base_name}_tokens.txt  ← un token por línea
    """
    os.makedirs(out_dir, exist_ok=True)

    clean_txt = " ".join(tokens)
    txt_path  = os.path.join(out_dir, f"{base_name}_limpia.txt")
    tok_path  = os.path.join(out_dir, f"{base_name}_tokens.txt")

    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(clean_txt)
        with open(tok_path, "w", encoding="utf-8") as f:
            f.write("\n".join(tokens))
        print(f"✅ Guardados:\n  • Texto limpio → {txt_path}\n  • Tokens → {tok_path}")
    except Exception as e:
        print(f"❌ Error al guardar archivos: {e}", file=sys.stderr)

if __name__ == "__main__":
    # 1) Lee la transcripción completa
    in_path = "data/transcripcion_completa.txt"
    try:
        with open(in_path, "r", encoding="utf-8") as f:
            texto_original = f.read()
    except FileNotFoundError:
        print(f"❌ No existe '{in_path}'", file=sys.stderr)
        sys.exit(1)

    # 2) Preprocesa
    tokens_procesados = preprocess_text(texto_original)
    
    clean_txt = " ".join(tokens_procesados)
    
    # 3) Corrige la ortografia 
    # tras obtener full_text
    full_text_corr = correct_text(clean_txt)
    print("🔧 Texto sin corrección ortográficamente:")
    print(clean_txt[:200], "…")
    print(25*"-")
    print("🔧 Texto corregido ortográficamente:")
    print(full_text_corr[:200], "…")

    
    # Uso tras generar clean_txt y tokens_procesados:
    pretty_print(clean_txt, tokens_procesados,
                char_preview=300,
                token_preview=12)

    # 3) Guarda tanto el texto limpio como los tokens
    save_transcription_and_tokens(
        raw_text=texto_original,
        tokens=tokens_procesados,
        out_dir="data",
        base_name="transcripcion"
    )
