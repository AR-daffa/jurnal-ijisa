"""
=============================================================
PREPROCESSING v2
Perubahan dari preprocessing.py versi awal:

1. Teks dasar sekarang 'combined_text_en' (title + abstract, sudah
   diterjemahkan ke Inggris oleh translate_to_english.py) — bukan lagi
   'title' Indonesia mentah. Otomatis menyelesaikan:
     - korpus title-only yang terlalu sparse (sekarang ada abstrak,
       untuk dokumen yang ketemu di Semantic Scholar)
     - keluaran topik campur ID/EN (sekarang semua sudah Inggris)

2. Karena teks sudah 1 bahasa (Inggris), stopword list & stemmer
   Bahasa Indonesia (Sastrawi) DIHAPUS — bukan cuma disederhanakan.
   Ini juga menutup celah yang kemarin ditemukan: daftar stopword
   ID/EN yang beda-beda dan gampang tidak sinkron antara file LDA
   dan file BERTopic. Sekarang cuma ADA SATU daftar stopword
   (Inggris), dipakai oleh preprocessing LDA di sini.

3. Nama kolom output (title_clean_lda, title_clean_bertopic) SENGAJA
   dipertahankan sama seperti sebelumnya, supaya lda_modeling.py dan
   bertopic_modeling.py tetap jalan tanpa perlu diubah.

FALLBACK: kalau file input masih yang lama (belum lewat
translate_to_english.py, tidak ada kolom combined_text_en), script ini
otomatis pakai kolom 'title' seperti sebelumnya, supaya tidak error.

REQUIREMENTS:
    pip install nltk pandas
    (Sastrawi TIDAK dibutuhkan lagi di versi ini)
=============================================================
"""

import re
import unicodedata
import pandas as pd

import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ═══════════════════════════════════════════════════════════
# STOPWORDS — SATU DAFTAR SAJA (Inggris), dipakai konsisten
# ═══════════════════════════════════════════════════════════

STOPWORDS_GENERIC_ACADEMIC = {
    # Kata generik akademik yang cenderung tidak informatif sebagai
    # keyword topik (BUKAN dibuang membabi buta — daftar ini sengaja
    # dijaga pendek, supaya istilah domain seperti "system"/"data" yang
    # sebelumnya ikut kebuang tidak otomatis hilang lagi).
    "study", "paper", "research", "approach", "method", "methods",
    "using", "based", "proposed", "case", "results", "new", "used",
}

ALL_STOPWORDS = set(stopwords.words("english")) | STOPWORDS_GENERIC_ACADEMIC
lemmatizer = WordNetLemmatizer()


# ═══════════════════════════════════════════════════════════
# FUNGSI PEMBERSIHAN
# ═══════════════════════════════════════════════════════════

def clean_text_basic(text: str) -> str:
    if not text or str(text).strip() in ["-", "nan", ""]:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = text.encode("ascii", "ignore").decode("utf-8")
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_for_lda(text: str) -> str:
    """lowercase → clean → tokenize → remove stopwords → lemmatize."""
    text = clean_text_basic(text)
    if not text:
        return ""
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in ALL_STOPWORDS and len(t) >= 3]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def preprocess_for_bertopic(text: str) -> str:
    """Minimal cleaning — pertahankan struktur kalimat untuk transformer."""
    if not text or str(text).strip() in ["-", "nan", ""]:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = text.encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^\w\s\.,\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ═══════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════

def run_preprocessing(input_path: str, output_path: str):
    print(f"\n{'='*55}")
    print("  PREPROCESSING v2 DIMULAI (English pipeline)")
    print(f"{'='*55}")

    df = pd.read_csv(input_path, encoding="utf-8")
    print(f"\n[INFO] Total artikel dimuat: {len(df)}")

    # ── Tentukan kolom teks dasar (dengan fallback) ─────────
    if "combined_text_en" in df.columns:
        base_col = "combined_text_en"
        print(f"[INFO] Memakai kolom '{base_col}' (title+abstract, EN)")
    else:
        base_col = "title"
        print(f"[WARN] Kolom 'combined_text_en' tidak ditemukan — "
              f"fallback ke kolom 'title' mentah. Jalankan "
              f"fetch_abstracts.py + translate_to_english.py dulu kalau "
              f"ingin korpus title+abstract berbahasa Inggris.")

    # ── 1. Hapus duplikat judul ──────────────────────────
    before = len(df)
    df.drop_duplicates(subset=["title"], keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"\n[1] Hapus duplikat: {before} -> {len(df)} artikel "
          f"(dihapus: {before - len(df)})")

    # ── 2. Hapus artikel tanpa judul ────────────────────
    df = df[df["title"].notna() & (df["title"] != "-")].copy()
    df.reset_index(drop=True, inplace=True)
    print(f"[2] Setelah filter judul kosong: {len(df)} artikel")

    # ── 3. Filter tahun ──────────────────────────────────
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    before = len(df)
    df = df[df["year"].between(2010, 2026) | df["year"].isna()].copy()
    df.reset_index(drop=True, inplace=True)
    print(f"[3] Filter tahun 2010-2026: {before} -> {len(df)} artikel")

    # ── 4. Preprocessing ────────────────────────────────
    print("\n[4] Memproses teks...")
    df["title_clean_lda"] = df[base_col].apply(preprocess_for_lda)
    df["title_clean_bertopic"] = df[base_col].apply(preprocess_for_bertopic)

    # ── 5. Hapus hasil kosong ────────────────────────────
    before = len(df)
    df = df[df["title_clean_lda"].str.len() > 0].copy()
    df.reset_index(drop=True, inplace=True)
    print(f"[5] Setelah hapus hasil preprocessing kosong: {before} -> {len(df)}")

    # ── 6. Statistik panjang token ───────────────────────
    df["token_count_lda"] = df["title_clean_lda"].apply(lambda x: len(x.split()))
    print(f"\n[STATISTIK] Panjang token LDA:"
          f"\n  - Rata-rata : {df['token_count_lda'].mean():.1f} token"
          f"\n  - Min       : {df['token_count_lda'].min()} token"
          f"\n  - Maks      : {df['token_count_lda'].max()} token")
    if "has_abstract" in df.columns:
        pct_abs = df["has_abstract"].sum() / len(df) * 100
        print(f"  - Dokumen dengan abstrak: {pct_abs:.1f}% "
              f"(sisanya title-only — laporkan angka ini di naskah)")

    # ── 7. Simpan hasil ──────────────────────────────────
    output_cols = [
        "author_id", "author_name", "title",
        "title_clean_lda", "title_clean_bertopic",
        "year", "cited_by", "token_count_lda",
    ]
    if "has_abstract" in df.columns:
        output_cols.append("has_abstract")

    df[output_cols].to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n[OK] Hasil disimpan ke: {output_path}")
    print(f"[OK] Total dokumen siap: {len(df)}")
    print(f"{'='*55}\n")

    return df


if __name__ == "__main__":
    # Pakai output translate_to_english.py kalau ada, kalau tidak
    # fallback otomatis ke semua_artikel.csv (title-only, ID)
    import os
    INPUT_FILE = ("semua_artikel_en.csv" if os.path.exists("semua_artikel_en.csv")
                  else "semua_artikel.csv")
    OUTPUT_FILE = "data_preprocessed.csv"

    df_result = run_preprocessing(INPUT_FILE, OUTPUT_FILE)