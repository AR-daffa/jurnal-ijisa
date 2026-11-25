import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings("ignore")

from gensim import corpora, models
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# ═══════════════════════════════════════════════════════════
# KONFIGURASI
# ═══════════════════════════════════════════════════════════

INPUT_FILE  = "data_preprocessed.csv"
OUTPUT_DIR  = "hasil_lda"
K_MIN       = 4
K_MAX       = 7
K_STEP      = 1
RANDOM_SEED = 42
PASSES      = 20
TOP_N_WORDS = 15


# ═══════════════════════════════════════════════════════════
# SEMUA KODE DIBUNGKUS if __name__ == '__main__'
# (WAJIB di Windows agar multiprocessing tidak error)
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'='*55}")
    print("  LDA TOPIC MODELING DIMULAI")
    print(f"{'='*55}")
    print(f"[INFO] Output akan disimpan di folder: {OUTPUT_DIR}/")

    # ─────────────────────────────────────────────────────
    # 1. LOAD DATA
    # ─────────────────────────────────────────────────────
    print("\n[1] Memuat data preprocessed...")
    df = pd.read_csv(INPUT_FILE, encoding="utf-8")
    print(f"    Total dokumen dimuat: {len(df)}")

    texts_raw = df["title_clean_lda"].dropna().tolist()
    texts_raw = [t for t in texts_raw if isinstance(t, str) and len(t.strip()) > 0]
    print(f"    Dokumen valid untuk LDA: {len(texts_raw)}")

    tokenized_texts = [text.split() for text in texts_raw]

    # IMPORT STOPWORDS LANGSUNG DARI SCRIPT PREPROCESSING
    from preprocessing import ALL_STOPWORDS

    # Filter stopwords langsung dari daftar token
    print(f"    [INFO] Menyaring {len(ALL_STOPWORDS)} stopwords...")
    tokenized_texts = [[word for word in text if word not in ALL_STOPWORDS] for text in tokenized_texts]

    # ─────────────────────────────────────────────────────
    # 2. DICTIONARY DAN CORPUS
    # ─────────────────────────────────────────────────────
    print("\n[2] Membangun dictionary dan corpus...")
    dictionary = corpora.Dictionary(tokenized_texts)

    before = len(dictionary)
    dictionary.filter_extremes(no_below=3, no_above=0.5)
    after = len(dictionary)
    print(f"    Kosakata sebelum filter : {before} kata")
    print(f"    Kosakata setelah filter : {after} kata")

    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    print(f"    Corpus: {len(corpus)} dokumen, {after} fitur kata")

    dictionary.save(f"{OUTPUT_DIR}/lda_dictionary.gensim")
    corpora.MmCorpus.serialize(f"{OUTPUT_DIR}/lda_corpus.mm", corpus)
    print(f"    Dictionary & corpus disimpan ke {OUTPUT_DIR}/")

    # ─────────────────────────────────────────────────────
    # TAMBAHAN: VERIFIKASI RATA-RATA TOKEN
    # ─────────────────────────────────────────────────────
    # 1. Rata-rata token asli (sebelum dictionary di-filter)
    total_tokens_raw = sum(len(text) for text in tokenized_texts)
    avg_tokens_raw = total_tokens_raw / len(tokenized_texts)
    
    # 2. Rata-rata token riil yang masuk ke model LDA (setelah filter_extremes)
    total_tokens_corpus = sum(sum(count for word_id, count in doc) for doc in corpus)
    avg_tokens_corpus = total_tokens_corpus / len(corpus)
    
    print("\n[VERIFIKASI] Analisis Panjang Token:")
    print(f"    Rata-rata token per dokumen (Sebelum Filter): {avg_tokens_raw:.2f} token/dokumen")
    print(f"    Rata-rata token per dokumen (Setelah Filter) : {avg_tokens_corpus:.2f} token/dokumen")

    # ─────────────────────────────────────────────────────
    # 3. CARI K OPTIMAL
    # ─────────────────────────────────────────────────────
    # ── [PERBAIKAN] Penyesuaian K_MAX dinamis berdasarkan ukuran korpus ──
    n_docs = len(tokenized_texts)
    # Batasi K_MAX agar tidak melebihi 10% dari total dokumen (minimal rentang sampai 3)
    dynamic_k_max = min(K_MAX, max(3, n_docs // 10))
    
    print(f"\n[3] Mencari K optimal (K={K_MIN} hingga K={dynamic_k_max} berdasarkan total {n_docs} dokumen)...")
    print("    Ini mungkin memakan waktu beberapa menit...\n")

    coherence_scores = []
    k_values = list(range(K_MIN, dynamic_k_max + 1, K_STEP))  # <-- Menggunakan dynamic_k_max

    for k in k_values:
        lda_model = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            random_state=RANDOM_SEED,
            passes=PASSES,
            alpha="auto",
            eta="auto",
            per_word_topics=True,
        )
        coherence_model = CoherenceModel(
            model=lda_model,
            texts=tokenized_texts,
            dictionary=dictionary,
            coherence="c_v",
            processes=1,          # paksa single process (fix Windows)
        )
        score = coherence_model.get_coherence()
        coherence_scores.append(score)
        print(f"    K={k:2d}  →  Coherence (Cv) = {score:.4f}")

    best_k_idx = coherence_scores.index(max(coherence_scores))
    best_k     = k_values[best_k_idx]
    best_score = coherence_scores[best_k_idx]
    print(f"\n    ✓ K OPTIMAL = {best_k}  (Cv = {best_score:.4f})")

    # Plot coherence vs K
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, coherence_scores, marker="o", linewidth=2,
             color="#2E75B6", markerfacecolor="#FF6B35", markersize=8)
    plt.axvline(x=best_k, color="red", linestyle="--", linewidth=1.5,
                label=f"Optimal K = {best_k}")
    plt.xlabel("Number of Topics (K)", fontsize=12)
    plt.ylabel("Topic Coherence (Cv)", fontsize=12)
    plt.title("Coherence Score vs Number of LDA Topics", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/coherence_vs_k.png", dpi=150)
    plt.close()
    print(f"    Grafik disimpan: {OUTPUT_DIR}/coherence_vs_k.png")

    df_coherence = pd.DataFrame({"K": k_values, "coherence_cv": coherence_scores})
    df_coherence.to_csv(f"{OUTPUT_DIR}/coherence_scores.csv", index=False)

    # ─────────────────────────────────────────────────────
    # 4. TRAINING MODEL FINAL
    # ─────────────────────────────────────────────────────
    print(f"\n[4] Training model LDA final dengan K={best_k}...")
    lda_final = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=best_k,
        random_state=RANDOM_SEED,
        passes=PASSES,
        alpha="auto",
        eta="auto",
        per_word_topics=True,
    )
    lda_final.save(f"{OUTPUT_DIR}/lda_model_k{best_k}.gensim")
    print(f"    Model disimpan: {OUTPUT_DIR}/lda_model_k{best_k}.gensim")

    # ─────────────────────────────────────────────────────
    # 5. EKSTRAK TOPIK
    # ─────────────────────────────────────────────────────
    print(f"\n[5] Topik yang ditemukan (K={best_k}):\n")
    topics_data = []
    for i in range(best_k):
        top_words  = lda_final.show_topic(i, topn=TOP_N_WORDS)
        words_str  = ", ".join([f"{w}({p:.3f})" for w, p in top_words])
        words_only = ", ".join([w for w, p in top_words])
        print(f"  Topik {i+1:2d}: {words_only}")
        topics_data.append({
            "topic_id"           : i + 1,
            "top_words"          : words_only,
            "top_words_with_prob": words_str,
            "label"              : f"Topik {i+1}",
        })

    df_topics = pd.DataFrame(topics_data)
    df_topics.to_csv(f"{OUTPUT_DIR}/lda_topics.csv", index=False, encoding="utf-8")
    print(f"\n    Tabel topik disimpan: {OUTPUT_DIR}/lda_topics.csv")

    # ─────────────────────────────────────────────────────
    # 6. DISTRIBUSI TOPIK PER DOKUMEN
    # ─────────────────────────────────────────────────────
    print(f"\n[6] Menghitung distribusi topik per dokumen...")
    doc_topics = []
    for i, bow in enumerate(corpus):
        topic_dist = lda_final.get_document_topics(bow, minimum_probability=0)
        dominant   = max(topic_dist, key=lambda x: x[1])
        doc_topics.append({
            "doc_idx"       : i,
            "dominant_topic": dominant[0] + 1,
            "topic_prob"    : round(dominant[1], 4),
        })

    df_doc_topics = pd.DataFrame(doc_topics)
    df_valid = df[df["title_clean_lda"].notna() &
                  (df["title_clean_lda"].str.strip() != "")].reset_index(drop=True)
    df_result = pd.concat([df_valid, df_doc_topics], axis=1)
    df_result.to_csv(f"{OUTPUT_DIR}/dokumen_dengan_topik.csv",
                     index=False, encoding="utf-8")
    print(f"    Disimpan: {OUTPUT_DIR}/dokumen_dengan_topik.csv")

    # Bar chart distribusi
    topic_counts = df_result["dominant_topic"].value_counts().sort_index()
    plt.figure(figsize=(max(8, best_k), 5))
    bars = plt.bar(topic_counts.index, topic_counts.values,
                   color="#2E75B6", edgecolor="white", linewidth=0.7)
    for bar, val in zip(bars, topic_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(val), ha="center", va="bottom", fontsize=9)
    plt.xlabel("Topic Number", fontsize=12)
    plt.ylabel("Number of Documents", fontsize=12)
    plt.title("Document Distribution per LDA Topic", fontsize=14, fontweight="bold")
    plt.xticks(range(1, best_k + 1))
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/distribusi_topik.png", dpi=150)
    plt.close()
    print(f"    Grafik distribusi disimpan: {OUTPUT_DIR}/distribusi_topik.png")

    # ─────────────────────────────────────────────────────
    # 7. ANALISIS TEMPORAL
    # ─────────────────────────────────────────────────────
    print(f"\n[7] Analisis temporal topik per tahun...")
    df_temporal = df_result[df_result["year"].notna()].copy()
    df_temporal["year"] = df_temporal["year"].astype(int)

    temporal_pivot = (
        df_temporal.groupby(["year", "dominant_topic"])
        .size().reset_index(name="count")
    )
    year_total = df_temporal.groupby("year").size().reset_index(name="total")
    temporal_pivot = temporal_pivot.merge(year_total, on="year")
    temporal_pivot["proportion"] = temporal_pivot["count"] / temporal_pivot["total"]
    temporal_pivot.to_csv(f"{OUTPUT_DIR}/temporal_topik.csv", index=False)

    top5_topics = topic_counts.head(5).index.tolist()
    colors = ["#2E75B6", "#FF6B35", "#28A745", "#FFC107", "#9B59B6"]
    plt.figure(figsize=(12, 6))
    for idx, topic_num in enumerate(top5_topics):
        data = temporal_pivot[
            temporal_pivot["dominant_topic"] == topic_num
        ].sort_values("year")
        plt.plot(data["year"], data["proportion"],
                 marker="o", linewidth=2,
                 label=f"Topic {topic_num}", color=colors[idx % len(colors)])
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Document Proportion", fontsize=12)
    plt.title("LDA Topic Trends per Year (Top 5 Largest Topics)",
              fontsize=14, fontweight="bold")
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/tren_topik_per_tahun.png", dpi=150)
    plt.close()
    print(f"    Grafik tren disimpan: {OUTPUT_DIR}/tren_topik_per_tahun.png")

    # ─────────────────────────────────────────────────────
    # 8. PYLDAVIS
    # ─────────────────────────────────────────────────────
    print(f"\n[8] Membuat visualisasi pyLDAvis...")
    try:
        vis_data = gensimvis.prepare(
            lda_final, corpus, dictionary, sort_topics=False
        )
        pyLDAvis.save_html(vis_data, f"{OUTPUT_DIR}/ldavis.html")
        print(f"    ✓ Visualisasi interaktif: {OUTPUT_DIR}/ldavis.html")
        print(f"    → Buka file ini di browser untuk eksplorasi topik!")
    except Exception as e:
        print(f"    [WARN] pyLDAvis gagal: {e}")

    # ─────────────────────────────────────────────────────
    # RINGKASAN AKHIR
    # ─────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  HASIL LDA — RINGKASAN")
    print(f"{'='*55}")
    print(f"  Jumlah dokumen  : {len(corpus)}")
    print(f"  Ukuran kosakata : {len(dictionary)} kata")
    print(f"  K optimal       : {best_k} topik")
    print(f"  Coherence (Cv)  : {best_score:.4f}")
    print(f"\n  File output di folder '{OUTPUT_DIR}/':")
    print(f"    coherence_vs_k.png       → grafik Cv vs K")
    print(f"    coherence_scores.csv     → tabel skor per K")
    print(f"    lda_topics.csv           → daftar topik + kata kunci")
    print(f"    dokumen_dengan_topik.csv → tiap artikel + topik dominan")
    print(f"    distribusi_topik.png     → bar chart jumlah dok/topik")
    print(f"    tren_topik_per_tahun.png → line chart tren temporal")
    print(f"    temporal_topik.csv       → data tren temporal")
    print(f"    ldavis.html              → visualisasi interaktif")
    print(f"\n  LANGKAH SELANJUTNYA:")
    print(f"  1. Buka lda_topics.csv → beri label manual tiap topik")
    print(f"  2. Buka ldavis.html di browser → eksplorasi topik")
    print(f"  3. Lanjut ke pemodelan BERTopic")
    print(f"{'='*55}\n")