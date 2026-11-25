"""
=============================================================
KOMPARASI LDA vs BERTopic (VERSI INTEGRASI FAIR COHERENCE)
Proposal Hibah 2026 - LDA vs BERTopic
=============================================================
REQUIREMENTS:
    pip install pandas numpy matplotlib seaborn gensim
    (bertopic & lda sudah harus dijalankan dulu)
=============================================================
JALANKAN SETELAH:
    1. lda_modeling.py      → hasil_lda/
    2. bertopic_modeling.py → hasil_bertopic/
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os
import json
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════
# KONFIGURASI
# ═══════════════════════════════════════════════════════════

LDA_DIR      = "hasil_lda"
BERTOPIC_DIR = "hasil_bertopic"
OUTPUT_DIR   = "hasil_komparasi"
TOP_N_WORDS  = 15

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n{'='*60}")
print("  KOMPARASI LDA vs BERTOPIC (FAIR COHERENCE)")
print(f"{'='*60}")

# ─────────────────────────────────────────────────────────
# 1. LOAD HASIL LDA
# ─────────────────────────────────────────────────────────
print("\n[1] Memuat hasil LDA...")

df_lda_topics = pd.read_csv(f"{LDA_DIR}/lda_topics.csv", encoding="utf-8")
df_lda_docs   = pd.read_csv(f"{LDA_DIR}/dokumen_dengan_topik.csv", encoding="utf-8")
df_lda_coh    = pd.read_csv(f"{LDA_DIR}/coherence_scores.csv", encoding="utf-8")

best_lda_coh = df_lda_coh["coherence_cv"].max()
best_lda_k   = int(df_lda_coh.loc[df_lda_coh["coherence_cv"].idxmax(), "K"])

all_words_lda = []
for words_str in df_lda_topics["top_words"].dropna():
    words = [w.strip() for w in words_str.split(",")]
    all_words_lda.extend(words)
lda_diversity = len(set(all_words_lda)) / len(all_words_lda) if all_words_lda else 0

print(f"    LDA  → K={best_lda_k}, Coherence(Cv, native)={best_lda_coh:.4f}, Diversity={lda_diversity:.4f}")

# ─────────────────────────────────────────────────────────
# 2. LOAD HASIL BERTOPIC + HITUNG FAIR COHERENCE
# ─────────────────────────────────────────────────────────
print("\n[2] Memuat hasil BERTopic & Menghitung Fair Coherence...")

df_bert_topics = pd.read_csv(f"{BERTOPIC_DIR}/bertopic_topics.csv", encoding="utf-8")
df_bert_docs   = pd.read_csv(f"{BERTOPIC_DIR}/dokumen_dengan_topik_bertopic.csv", encoding="utf-8")

with open(f"{BERTOPIC_DIR}/bertopic_metrics.json", encoding="utf-8") as f:
    bert_metrics = json.load(f)

bert_n_topics  = bert_metrics["n_topics"]
bert_diversity = bert_metrics["topic_diversity"]
bert_outlier   = bert_metrics["outlier_pct"]

# ── Fair coherence: LDA & BERTopic dievaluasi pada korpus referensi
#    yang SAMA (data_preprocessed.csv lengkap, bukan korpus training
#    masing-masing model) — lihat fair_coherence.py untuk penjelasan.
from fair_coherence import compute_fair_coherence

df_full_preprocessed = pd.read_csv("data_preprocessed.csv", encoding="utf-8")
reference_texts = df_full_preprocessed["title_clean_bertopic"].dropna().tolist()

def _to_word_lists(df_topics, top_n=TOP_N_WORDS):
    out = []
    for words_str in df_topics["top_words"].dropna():
        words = [w.strip() for w in str(words_str).split(",")][:top_n]
        out.append(words)
    return out

lda_topic_words  = _to_word_lists(df_lda_topics)
bert_topic_words = _to_word_lists(df_bert_topics)

fair = compute_fair_coherence(lda_topic_words, bert_topic_words, reference_texts)
lda_coherence_fair  = fair["lda_coherence_fair"]
bert_coherence_fair = fair["bert_coherence_fair"]

print(f"    [Fair, korpus referensi sama, n={fair['n_reference_docs']} dok]")
print(f"    LDA  → Coherence(Cv, fair)={lda_coherence_fair:.4f}")
print(f"    BERTopic → K={bert_n_topics}, Coherence(Cv, fair)={bert_coherence_fair:.4f}, "
      f"Diversity={bert_diversity:.4f}, Outlier={bert_outlier:.1f}%")

# Skor "native" (dari korpus training masing-masing model) tetap
# disimpan sebagai referensi/pembanding transparansi di naskah,
# TAPI yang dipakai untuk klaim head-to-head adalah versi "fair" di bawah.
bert_coherence = bert_coherence_fair   # override supaya kode Bagian 3-4 di bawah tetap jalan
best_lda_coh_native = best_lda_coh     # simpan skor native LDA sebelum di-override
best_lda_coh = lda_coherence_fair      # override supaya tabel & grafik pakai versi fair

# ─────────────────────────────────────────────────────────
# 3. TABEL PERBANDINGAN KUANTITATIF
# ─────────────────────────────────────────────────────────
print("\n[3] Membuat tabel perbandingan kuantitatif...")

n_docs_lda  = len(df_lda_docs)
n_docs_bert = len(df_bert_docs[df_bert_docs["bertopic_topic"] != -1])

comparison = {
    "Metrik": [
        "Jumlah Topik",
        "Topic Coherence (Cv, Fair)",
        "Topic Diversity",
        "Dokumen Terklasifikasi",
        "Dokumen Outlier (%)",
        "Pendekatan Representasi",
        "Penanganan Teks Pendek",
        "Interpretabilitas Topik",
        "Kecepatan Training",
        "Kebutuhan GPU",
    ],
    "LDA": [
        best_lda_k,
        f"{best_lda_coh:.4f}",
        f"{lda_diversity:.4f}",
        f"{n_docs_lda} (100%)",
        "0% (semua terklasifikasi)",
        "Bag-of-Words (BoW)",
        "Terbatas (sparse data)",
        "Tinggi (statistik)",
        "Cepat",
        "Tidak",
    ],
    "BERTopic": [
        bert_n_topics,
        f"{bert_coherence:.4f}" if bert_coherence else "N/A",
        f"{bert_diversity:.4f}",
        f"{n_docs_bert} ({100 - bert_outlier:.1f}%)",
        f"{bert_outlier:.1f}%",
        "Transformer Embeddings",
        "Baik (kontekstual)",
        "Sedang (perlu interpretasi)",
        "Lebih lambat",
        "Opsional (CPU bisa)",
    ],
}

df_comparison = pd.DataFrame(comparison)
df_comparison.to_csv(f"{OUTPUT_DIR}/tabel_perbandingan.csv", index=False, encoding="utf-8")
print(f"    Tabel disimpan: {OUTPUT_DIR}/tabel_perbandingan.csv")
print(f"\n{df_comparison.to_string(index=False)}")

# ─────────────────────────────────────────────────────────
# 4. VISUALISASI PERBANDINGAN METRIK
# ─────────────────────────────────────────────────────────
print(f"\n[4] Membuat visualisasi perbandingan metrik...")

colors_lda  = "#2E75B6"
colors_bert = "#28A745"

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("Quantitative Comparison: LDA vs BERTopic (Fair Evaluation)",
             fontsize=16, fontweight="bold", y=1.02)

# Coherence
ax = axes[0]
coh_values = [best_lda_coh, bert_coherence if bert_coherence else 0]
bars = ax.bar(["LDA", "BERTopic"], coh_values,
              color=[colors_lda, colors_bert], width=0.5, edgecolor="white")
for bar, val in zip(bars, coh_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.4f}" if val else "N/A",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_title("Topic Coherence (Cv, Fair)", fontsize=13, fontweight="bold")
ax.set_ylabel("Coherence Score", fontsize=11)
ax.set_ylim(0, max(coh_values) * 1.3 if any(coh_values) else 1)
ax.grid(axis="y", alpha=0.3)

# Diversity
ax = axes[1]
div_values = [lda_diversity, bert_diversity]
bars = ax.bar(["LDA", "BERTopic"], div_values,
              color=[colors_lda, colors_bert], width=0.5, edgecolor="white")
for bar, val in zip(bars, div_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_title("Topic Diversity", fontsize=13, fontweight="bold")
ax.set_ylabel("Diversity Score (0–1)", fontsize=11)
ax.set_ylim(0, 1.15)
ax.grid(axis="y", alpha=0.3)

# Jumlah Topik
ax = axes[2]
topic_vals = [best_lda_k, bert_n_topics]
bars = ax.bar(["LDA", "BERTopic"], topic_vals,
              color=[colors_lda, colors_bert], width=0.5, edgecolor="white")
for bar, val in zip(bars, topic_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_title("Number of Topics", fontsize=13, fontweight="bold")
ax.set_ylabel("Number of Topics (K)", fontsize=11)
ax.set_ylim(0, max(topic_vals) * 1.3)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/komparasi_metrik.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"    Grafik metrik: {OUTPUT_DIR}/komparasi_metrik.png")

# ─────────────────────────────────────────────────────────
# 5. OVERLAY TREN TEMPORAL
# ─────────────────────────────────────────────────────────
print(f"\n[5] Membuat grafik tren temporal overlay...")

try:
    df_lda_temporal  = pd.read_csv(f"{LDA_DIR}/temporal_topik.csv")
    df_bert_temporal = pd.read_csv(f"{BERTOPIC_DIR}/temporal_topik_bertopic.csv")

    lda_agg  = df_lda_temporal.groupby("year")["count"].sum().reset_index()
    bert_agg = df_bert_temporal.groupby("year")["count"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(lda_agg["year"], lda_agg["count"],
            marker="o", linewidth=2.5, color=colors_lda, label="LDA")
    ax.plot(bert_agg["year"], bert_agg["count"],
            marker="s", linewidth=2.5, color=colors_bert,
            linestyle="--", label="BERTopic")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Number of Classified Documents", fontsize=12)
    ax.set_title("Temporal Trends: LDA vs BERTopic",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/tren_temporal_overlay.png", dpi=150)
    plt.close()
    print(f"    Grafik temporal: {OUTPUT_DIR}/tren_temporal_overlay.png")
except Exception as e:
    print(f"    [WARN] Grafik temporal gagal: {e}")

# ─────────────────────────────────────────────────────────
# 6. ANALISIS LIFE CYCLE TOPIK
# ─────────────────────────────────────────────────────────
print(f"\n[6] Analisis Life Cycle Topik (Emerging / Growth / Maturity / Declining)...")

def classify_lifecycle(temporal_df, topic_col, year_col="year", count_col="count"):
    results = []
    max_year     = temporal_df[year_col].max()
    recent_years = [max_year - 2, max_year - 1, max_year]

    for topic in temporal_df[topic_col].unique():
        data = temporal_df[temporal_df[topic_col] == topic].sort_values(year_col)
        total_docs   = data[count_col].sum()
        avg_per_year = data[count_col].mean()

        recent = data[data[year_col].isin(recent_years)]
        if len(recent) >= 2:
            rs = recent.sort_values(year_col)
            growth_rate = (rs[count_col].iloc[-1] - rs[count_col].iloc[0]) / (rs[count_col].iloc[0] + 1)
        else:
            growth_rate = 0

        first_year = data[year_col].min()
        age        = max_year - first_year

        if age <= 2 and growth_rate >= 0:
            phase = "Emerging"
        elif growth_rate > 0.3:
            phase = "Growth"
        elif growth_rate < -0.2:
            phase = "Declining"
        else:
            phase = "Maturity"

        results.append({
            topic_col     : topic,
            "total_docs"  : int(total_docs),
            "avg_per_year": round(avg_per_year, 1),
            "growth_rate" : round(growth_rate, 3),
            "first_year"  : int(first_year),
            "topic_age"   : int(age),
            "phase"       : phase,
        })

    return pd.DataFrame(results).sort_values("total_docs", ascending=False)


df_lda_lifecycle  = None
df_bert_lifecycle = None

try:
    df_lda_temporal  = pd.read_csv(f"{LDA_DIR}/temporal_topik.csv")
    df_lda_lifecycle = classify_lifecycle(df_lda_temporal, "dominant_topic")
    df_lda_lifecycle = df_lda_lifecycle.merge(
        df_lda_topics[["topic_id", "top_words"]].rename(
            columns={"topic_id": "dominant_topic", "top_words": "kata_kunci"}),
        on="dominant_topic", how="left"
    )
    df_lda_lifecycle.to_csv(f"{OUTPUT_DIR}/lifecycle_topik_lda.csv",
                             index=False, encoding="utf-8")
    print(f"    LDA Life Cycle:")
    for phase in ["Emerging", "Growth", "Maturity", "Declining"]:
        n = (df_lda_lifecycle["phase"] == phase).sum()
        print(f"      {phase:10s}: {n} topik")
except Exception as e:
    print(f"    [WARN] LDA lifecycle gagal: {e}")

try:
    df_bert_temporal  = pd.read_csv(f"{BERTOPIC_DIR}/temporal_topik_bertopic.csv")
    df_bert_lifecycle = classify_lifecycle(df_bert_temporal, "bertopic_topic")
    df_bert_lifecycle = df_bert_lifecycle.merge(
        df_bert_topics[["topic_id", "top_words"]].rename(
            columns={"topic_id": "bertopic_topic", "top_words": "kata_kunci"}),
        on="bertopic_topic", how="left"
    )
    df_bert_lifecycle.to_csv(f"{OUTPUT_DIR}/lifecycle_topik_bertopic.csv",
                              index=False, encoding="utf-8")
    print(f"    BERTopic Life Cycle:")
    for phase in ["Emerging", "Growth", "Maturity", "Declining"]:
        n = (df_bert_lifecycle["phase"] == phase).sum()
        print(f"      {phase:10s}: {n} topik")
except Exception as e:
    print(f"    [WARN] BERTopic lifecycle gagal: {e}")

# ─────────────────────────────────────────────────────────
# 7. BUBBLE CHART LIFE CYCLE
# ─────────────────────────────────────────────────────────
print(f"\n[7] Membuat bubble chart life cycle...")

phase_colors = {
    "Emerging" : "#FF6B35",
    "Growth"   : "#28A745",
    "Maturity" : "#2E75B6",
    "Declining": "#ADB5BD",
}

for model_name, df_lc, topic_col in [
    ("LDA",      df_lda_lifecycle,  "dominant_topic"),
    ("BERTopic", df_bert_lifecycle, "bertopic_topic"),
]:
    if df_lc is None or df_lc.empty:
        continue
    try:
        fig, ax = plt.subplots(figsize=(12, 7))
        for _, row in df_lc.iterrows():
            color = phase_colors.get(row["phase"], "#999")
            size  = max(100, row["total_docs"] * 20)
            ax.scatter(row["first_year"], row["growth_rate"],
                       s=size, color=color, alpha=0.7,
                       edgecolors="white", linewidth=1.5)
            ax.annotate(f"T{row[topic_col]}",
                        (row["first_year"], row["growth_rate"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)

        patches = [mpatches.Patch(color=c, label=p) for p, c in phase_colors.items()]
        ax.legend(handles=patches, title="Phase", fontsize=10, title_fontsize=11)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xlabel("Year of First Appearance", fontsize=12)
        ax.set_ylabel("Growth Rate (Last 3 Years)", fontsize=12)
        ax.set_title(f"{model_name} Topic Life Cycle (Size = Document Count)",
                     fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fname = f"{OUTPUT_DIR}/lifecycle_bubble_{model_name.lower()}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"    Bubble chart {model_name}: {fname}")
    except Exception as e:
        print(f"    [WARN] Bubble chart {model_name} gagal: {e}")

# ─────────────────────────────────────────────────────────
# 8. REKOMENDASI ROADMAP
# ─────────────────────────────────────────────────────────
print(f"\n[8] Membuat rekomendasi roadmap...")

rekomendasi_map = {
    "Emerging" : "Prioritaskan eksplorasi & kolaborasi awal",
    "Growth"   : "Perkuat dengan hibah & dorong publikasi",
    "Maturity" : "Konsolidasi & integrasikan ke kurikulum",
    "Declining": "Evaluasi relevansi, pertimbangkan pivot topik",
}

roadmap_rows = []
for model_name, df_lc, topic_col, word_col in [
    ("LDA",      df_lda_lifecycle,  "dominant_topic", "kata_kunci"),
    ("BERTopic", df_bert_lifecycle, "bertopic_topic", "kata_kunci"),
]:
    if df_lc is None or df_lc.empty:
        continue
    for phase in ["Emerging", "Growth", "Maturity", "Declining"]:
        subset = df_lc[df_lc["phase"] == phase].head(3)
        for _, row in subset.iterrows():
            roadmap_rows.append({
                "model"      : model_name,
                "fase"       : phase,
                "topik_id"   : row[topic_col],
                "total_dok"  : row["total_docs"],
                "growth_rate": row["growth_rate"],
                "kata_kunci" : str(row.get(word_col, ""))[:80],
                "rekomendasi": rekomendasi_map.get(phase, "-"),
            })

df_roadmap = pd.DataFrame(roadmap_rows)
df_roadmap.to_csv(f"{OUTPUT_DIR}/rekomendasi_roadmap.csv", index=False, encoding="utf-8")
print(f"    Rekomendasi roadmap: {OUTPUT_DIR}/rekomendasi_roadmap.csv")

# ─────────────────────────────────────────────────────────
# RINGKASAN AKHIR
# ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  KOMPARASI SELESAI — RINGKASAN OUTPUT")
print(f"{'='*60}")
print(f"  Folder output : {OUTPUT_DIR}/")
print(f"")
print(f"  tabel_perbandingan.csv           → metrik LDA vs BERTopic → Bab IV")
print(f"  komparasi_metrik.png             → grafik 3 metrik → Bab IV")
print(f"  tren_temporal_overlay.png        → tren overlay → Bab IV")
print(f"  lifecycle_topik_lda.csv          → fase topik LDA → Bab IV/V")
print(f"  lifecycle_topik_bertopic.csv     → fase topik BERTopic → Bab IV/V")
print(f"  lifecycle_bubble_lda.png         → bubble chart LDA → Bab V")
print(f"  lifecycle_bubble_bertopic.png    → bubble chart BERTopic → Bab V")
print(f"  rekomendasi_roadmap.csv          → basis narasi roadmap → Bab V")
print(f"")
print(f"  LANGKAH SELANJUTNYA:")
print(f"  1. Buka lifecycle_topik_*.csv  → beri label manual tiap topik")
print(f"  2. Buka rekomendasi_roadmap.csv → susun narasi Bab V proposal")
print(f"  3. Masukkan semua .png ke dokumen proposal Hibah2026.docx")
print(f"{'='*60}\n")