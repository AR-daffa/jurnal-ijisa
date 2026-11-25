"""
=============================================================
BERTOPIC MODELING
Proposal Hibah 2026 - LDA vs BERTopic
=============================================================
REQUIREMENTS:
    pip install bertopic sentence-transformers umap-learn hdbscan
    pip install matplotlib seaborn pandas scikit-learn plotly
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import json
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════
# KONFIGURASI
# ═══════════════════════════════════════════════════════════

INPUT_FILE     = "data_preprocessed.csv"
OUTPUT_DIR     = "hasil_bertopic"
LDA_DIR        = "hasil_lda"         # untuk perbandingan nanti
RANDOM_SEED    = 42
TOP_N_WORDS    = 15

# Model khusus Inggris yang ringan dan sangat akurat
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 

# UMAP params
UMAP_N_NEIGHBORS    = 15
UMAP_N_COMPONENTS   = 5
UMAP_MIN_DIST       = 0.0

# HDBSCAN params
HDBSCAN_MIN_CLUSTER = 5    # turunkan jika dokumen sedikit (<200)
HDBSCAN_MIN_SAMPLES = 3

# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'='*55}")
    print("  BERTOPIC MODELING DIMULAI")
    print(f"{'='*55}")
    print(f"[INFO] Embedding model : {EMBEDDING_MODEL}")
    print(f"[INFO] Output dir      : {OUTPUT_DIR}/")

    # ─────────────────────────────────────────────────────
    # 1. LOAD DATA
    # ─────────────────────────────────────────────────────
    print("\n[1] Memuat data preprocessed...")
    df = pd.read_csv(INPUT_FILE, encoding="utf-8")
    print(f"    Total dokumen dimuat: {len(df)}")

    # BERTopic pakai kolom title_clean_bertopic (minimal cleaning)
    docs_raw = df["title_clean_bertopic"].dropna().tolist()
    docs_raw = [d for d in docs_raw if isinstance(d, str) and len(d.strip()) > 5]
    print(f"    Dokumen valid untuk BERTopic: {len(docs_raw)}")

    # Simpan index valid agar bisa digabung kembali ke df
    df_valid = df[
        df["title_clean_bertopic"].notna() &
        (df["title_clean_bertopic"].str.strip().str.len() > 5)
    ].reset_index(drop=True)

    # ─────────────────────────────────────────────────────
    # 2. INISIALISASI KOMPONEN BERTOPIC
    # ─────────────────────────────────────────────────────
    print("\n[2] Menginisialisasi komponen BERTopic...")

    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer

    # Embedding model 
    print(f"    Loading embedding model: {EMBEDDING_MODEL} ...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    # UMAP untuk reduksi dimensi
    umap_model = UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=UMAP_N_COMPONENTS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=RANDOM_SEED,
    )

    # ── [PERBAIKAN] Penyesuaian HDBSCAN dinamis berdasarkan ukuran korpus ──
    n_docs = len(docs_raw)
    # Jika dokumen sedikit, min_cluster diturunkan otomatis (minimal 3)
    dynamic_min_cluster = max(3, n_docs // 30) 
    
    print(f"    [INFO] Total dokumen valid: {n_docs}")
    print(f"    [Dinamis] HDBSCAN min_cluster_size disesuaikan menjadi: {dynamic_min_cluster}")

    # HDBSCAN untuk clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size=dynamic_min_cluster,  # <-- Menggunakan nilai dinamis
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    # IMPORT STOPWORDS LANGSUNG DARI SCRIPT PREPROCESSING
    from preprocessing import ALL_STOPWORDS

    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    
    # Gabungkan stop words akademik dari preprocessing dengan stop words Inggris standard
    combined_stopwords = set(ALL_STOPWORDS).union(ENGLISH_STOP_WORDS)

    vectorizer_model = CountVectorizer(
        stop_words=list(combined_stopwords),  
        min_df=2,
        ngram_range=(1, 2),
    )

    # c-TF-IDF dengan BM25 weighting
    cttfidf_model = ClassTfidfTransformer(
        reduce_frequent_words=True,
        bm25_weighting=True,
    )

    print("    Semua komponen siap.")

    # ─────────────────────────────────────────────────────
    # 3. FIT BERTOPIC
    # ─────────────────────────────────────────────────────
    print("\n[3] Fitting BERTopic (ini mungkin 2–5 menit)...")

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=cttfidf_model,  # <-- SUDAH DIPERBAIKI
        top_n_words=TOP_N_WORDS,
        verbose=True,
        calculate_probabilities=True,
    )

    topics, probs = topic_model.fit_transform(docs_raw)
    print(f"\n    ✓ BERTopic selesai!")

    # [Tambahan] Optimasi Pengurangan Outlier (Topik -1)
    # Jalankan ini jika persentase outlier dirasa terlalu tinggi
    try:
        print("\n[OPTIMASI] Mengurangi jumlah outlier (Topic -1)...")
        # Distribusikan ulang dokumen outlier berdasarkan kesamaan c-TF-IDF
        new_topics = topic_model.reduce_outliers(docs_raw, topics, strategy="c-tf-idf")

        # PENTING: vectorizer_model & ctfidf_model harus di-pass ulang di sini,
        # kalau tidak, BERTopic diam-diam membangun CountVectorizer() default
        # (tanpa stopwords, tanpa min_df) untuk menghitung ulang top words —
        # inilah sumber "of"/"the"/"and" yang muncul di topic keywords.
        topic_model.update_topics(
            docs_raw,
            topics=new_topics,
            vectorizer_model=vectorizer_model,
            ctfidf_model=cttfidf_model,  # <-- SUDAH DIPERBAIKI
        )
        topics = new_topics

        # Update ulang topic_info setelah outlier diredistribusi
        topic_info = topic_model.get_topic_info()
    except Exception as e:
        print(f"    [WARN] Gagal melakukan reduce_outliers: {e}")

    # ─────────────────────────────────────────────────────
    # 4. STATISTIK TOPIK
    # ─────────────────────────────────────────────────────
    topic_info = topic_model.get_topic_info()
    n_topics    = len(topic_info[topic_info["Topic"] != -1])
    n_outliers  = topic_info[topic_info["Topic"] == -1]["Count"].values[0] if -1 in topic_info["Topic"].values else 0

    print(f"\n[4] Hasil BERTopic:")
    print(f"    Jumlah topik ditemukan : {n_topics}")
    print(f"    Dokumen outlier (-1)   : {n_outliers} ({n_outliers/len(docs_raw)*100:.1f}%)")
    print(f"    Dokumen terklasifikasi : {len(docs_raw) - n_outliers}")

    # Simpan topic_info
    topic_info.to_csv(f"{OUTPUT_DIR}/bertopic_topic_info.csv", index=False, encoding="utf-8")

    # ─────────────────────────────────────────────────────
    # 5. EKSTRAK KATA KUNCI TIAP TOPIK
    # ─────────────────────────────────────────────────────
    print(f"\n[5] Topik yang ditemukan:\n")
    topics_data = []
    for topic_id in sorted(topic_model.get_topics().keys()):
        if topic_id == -1:
            continue
        top_words = topic_model.get_topic(topic_id)
        words_only = ", ".join([w for w, _ in top_words[:TOP_N_WORDS]])
        words_prob = ", ".join([f"{w}({p:.3f})" for w, p in top_words[:TOP_N_WORDS]])
        count = topic_info[topic_info["Topic"] == topic_id]["Count"].values[0]
        print(f"  Topik {topic_id+1:2d} ({count:3d} dok): {words_only[:80]}")
        topics_data.append({
            "topic_id"           : topic_id + 1,
            "topic_id_raw"       : topic_id,
            "doc_count"          : int(count),
            "top_words"          : words_only,
            "top_words_with_prob": words_prob,
            "label"              : f"Topik {topic_id+1}",
        })

    df_topics = pd.DataFrame(topics_data)
    df_topics.to_csv(f"{OUTPUT_DIR}/bertopic_topics.csv", index=False, encoding="utf-8")
    print(f"\n    Tabel topik disimpan: {OUTPUT_DIR}/bertopic_topics.csv")

    # ─────────────────────────────────────────────────────
    # 6. DISTRIBUSI TOPIK PER DOKUMEN
    # ─────────────────────────────────────────────────────
    print(f"\n[6] Menyimpan distribusi topik per dokumen...")
    df_valid["bertopic_topic_raw"] = topics
    df_valid["bertopic_topic"]     = df_valid["bertopic_topic_raw"].apply(
        lambda x: x + 1 if x != -1 else -1   # mulai dari 1, outlier tetap -1
    )
    df_valid["bertopic_prob"] = [round(float(np.max(p)), 4) if hasattr(p, '__len__') else round(float(p), 4)
                                  for p in probs]
    df_valid.to_csv(f"{OUTPUT_DIR}/dokumen_dengan_topik_bertopic.csv",
                    index=False, encoding="utf-8")
    print(f"    Disimpan: {OUTPUT_DIR}/dokumen_dengan_topik_bertopic.csv")

    # Bar chart distribusi (tanpa outlier)
    df_plot = df_valid[df_valid["bertopic_topic"] != -1]
    topic_counts = df_plot["bertopic_topic"].value_counts().sort_index()

    plt.figure(figsize=(max(10, n_topics), 5))
    bars = plt.bar(topic_counts.index, topic_counts.values,
                   color="#28A745", edgecolor="white", linewidth=0.7)
    for bar, val in zip(bars, topic_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 str(val), ha="center", va="bottom", fontsize=8)
    plt.xlabel("Topic Number", fontsize=12)
    plt.ylabel("Number of Documents", fontsize=12)
    plt.title("Document Distribution per BERTopic Topic", fontsize=14, fontweight="bold")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/distribusi_topik_bertopic.png", dpi=150)
    plt.close()
    print(f"    Grafik distribusi disimpan: {OUTPUT_DIR}/distribusi_topik_bertopic.png")

    # ─────────────────────────────────────────────────────
    # 7. ANALISIS TEMPORAL
    # ─────────────────────────────────────────────────────
    print(f"\n[7] Analisis temporal topik per tahun...")
    df_temporal = df_valid[
        (df_valid["year"].notna()) & (df_valid["bertopic_topic"] != -1)
    ].copy()
    df_temporal["year"] = df_temporal["year"].astype(int)

    temporal_pivot = (
        df_temporal.groupby(["year", "bertopic_topic"])
        .size().reset_index(name="count")
    )
    year_total = df_temporal.groupby("year").size().reset_index(name="total")
    temporal_pivot = temporal_pivot.merge(year_total, on="year")
    temporal_pivot["proportion"] = temporal_pivot["count"] / temporal_pivot["total"]
    temporal_pivot.to_csv(f"{OUTPUT_DIR}/temporal_topik_bertopic.csv", index=False)

    top5_topics = topic_counts.head(5).index.tolist()
    colors = ["#28A745", "#2E75B6", "#FF6B35", "#FFC107", "#9B59B6"]
    plt.figure(figsize=(12, 6))
    for idx, topic_num in enumerate(top5_topics):
        data = temporal_pivot[
            temporal_pivot["bertopic_topic"] == topic_num
        ].sort_values("year")
        if len(data) < 2:
            continue
        plt.plot(data["year"], data["proportion"],
                 marker="o", linewidth=2,
                 label=f"Topic {topic_num}", color=colors[idx % len(colors)])
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Document Proportion", fontsize=12)
    plt.title("BERTopic Topic Trends per Year (Top 5 Largest Topics)",
              fontsize=14, fontweight="bold")
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/tren_topik_bertopic_per_tahun.png", dpi=150)
    plt.close()
    print(f"    Grafik tren disimpan: {OUTPUT_DIR}/tren_topik_bertopic_per_tahun.png")

    # ─────────────────────────────────────────────────────
    # 8. TOPIC DIVERSITY
    # ─────────────────────────────────────────────────────
    print(f"\n[8] Menghitung Topic Diversity...")

    all_top_words = []
    for topic_id in sorted(topic_model.get_topics().keys()):
        if topic_id == -1:
            continue
        words = [w for w, _ in topic_model.get_topic(topic_id)[:TOP_N_WORDS]]
        all_top_words.extend(words)

    unique_words = set(all_top_words)
    topic_diversity = len(unique_words) / len(all_top_words) if all_top_words else 0
    print(f"    Topic Diversity BERTopic : {topic_diversity:.4f}")

    # Simpan metrik
    metrics = {
        "model"           : "BERTopic",
        "embedding_model" : EMBEDDING_MODEL,
        "n_topics"        : n_topics,
        "n_outliers"      : int(n_outliers),
        "outlier_pct"     : round(n_outliers / len(docs_raw) * 100, 2),
        "topic_diversity" : round(topic_diversity, 4),
        "top_n_words"     : TOP_N_WORDS,
    }
    with open(f"{OUTPUT_DIR}/bertopic_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"    Metrik disimpan: {OUTPUT_DIR}/bertopic_metrics.json")

    # ─────────────────────────────────────────────────────
    # 9. VISUALISASI INTERAKTIF (HTML)
    # ─────────────────────────────────────────────────────
    print(f"\n[9] Membuat visualisasi interaktif...")

    try:
        # Intertopic distance map
        fig_topics = topic_model.visualize_topics()
        fig_topics.write_html(f"{OUTPUT_DIR}/bertopic_intertopic_map.html")
        print(f"    ✓ Intertopic distance map: {OUTPUT_DIR}/bertopic_intertopic_map.html")
    except Exception as e:
        print(f"    [WARN] Intertopic map gagal: {e}")

    try:
        # Bar chart kata kunci per topik
        fig_barchart = topic_model.visualize_barchart(top_n_topics=min(n_topics, 16), n_words=10)
        fig_barchart.write_html(f"{OUTPUT_DIR}/bertopic_barchart.html")
        print(f"    ✓ Barchart kata kunci: {OUTPUT_DIR}/bertopic_barchart.html")
    except Exception as e:
        print(f"    [WARN] Barchart gagal: {e}")

    try:
        # Topic hierarchy
        fig_hierarchy = topic_model.visualize_hierarchy()
        fig_hierarchy.write_html(f"{OUTPUT_DIR}/bertopic_hierarchy.html")
        print(f"    ✓ Topic hierarchy: {OUTPUT_DIR}/bertopic_hierarchy.html")
    except Exception as e:
        print(f"    [WARN] Hierarchy gagal: {e}")

    try:
        # Heatmap similarity antar topik
        fig_heatmap = topic_model.visualize_heatmap()
        fig_heatmap.write_html(f"{OUTPUT_DIR}/bertopic_heatmap.html")
        print(f"    ✓ Topic heatmap: {OUTPUT_DIR}/bertopic_heatmap.html")
    except Exception as e:
        print(f"    [WARN] Heatmap gagal: {e}")

    # ─────────────────────────────────────────────────────
    # 10. SIMPAN MODEL
    # ─────────────────────────────────────────────────────
    print(f"\n[10] Menyimpan model BERTopic...")
    try:
        topic_model.save(f"{OUTPUT_DIR}/bertopic_model",
                         serialization="safetensors",
                         save_ctfidf=True,
                         save_embedding_model=EMBEDDING_MODEL)
        print(f"     Model disimpan: {OUTPUT_DIR}/bertopic_model/")
    except Exception as e:
        print(f"     [WARN] Save model gagal ({e}), coba format pickle...")
        import pickle
        with open(f"{OUTPUT_DIR}/bertopic_model.pkl", "wb") as f:
            pickle.dump(topic_model, f)
        print(f"     Model disimpan: {OUTPUT_DIR}/bertopic_model.pkl")

    # ─────────────────────────────────────────────────────
    # RINGKASAN AKHIR
    # ─────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  HASIL BERTOPIC — RINGKASAN")
    print(f"{'='*55}")
    print(f"  Jumlah dokumen      : {len(docs_raw)}")
    print(f"  Jumlah topik        : {n_topics}")
    print(f"  Dokumen outlier     : {n_outliers} ({n_outliers/len(docs_raw)*100:.1f}%)")
    print(f"  Topic Diversity     : {topic_diversity:.4f}")
    print(f"\n  File output di folder '{OUTPUT_DIR}/':")
    print(f"    bertopic_topic_info.csv           → info semua topik")
    print(f"    bertopic_topics.csv               → kata kunci per topik")
    print(f"    dokumen_dengan_topik_bertopic.csv → artikel + topik dominan")
    print(f"    distribusi_topik_bertopic.png     → bar chart distribusi")
    print(f"    tren_topik_bertopic_per_tahun.png → tren temporal")
    print(f"    temporal_topik_bertopic.csv       → data temporal")
    print(f"    bertopic_metrics.json             → metrik evaluasi")
    print(f"    bertopic_intertopic_map.html     → peta jarak antar topik")
    print(f"    bertopic_barchart.html           → kata kunci interaktif")
    print(f"    bertopic_hierarchy.html          → hierarki topik")
    print(f"    bertopic_heatmap.html            → similarity antar topik")
    print(f"\n  LANGKAH SELANJUTNYA:")
    print(f"  1. Buka bertopic_topics.csv → beri label manual tiap topik")
    print(f"  2. Buka HTML files di browser → eksplorasi visual")
    print(f"  3. Jalankan komparasi_lda_bertopic.py untuk analisis perbandingan")
    print(f"{'='*55}\n")