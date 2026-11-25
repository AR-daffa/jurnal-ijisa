"""
=============================================================
GENERATE LAPORAN BAB IV & V
Proposal Hibah 2026 - LDA vs BERTopic
=============================================================
Script ini membuat dokumen Word berisi:
  - Bab IV: Hasil Penelitian (otomatis dari CSV/JSON output)
  - Bab V: Ringkasan Pembahasan & Roadmap
=============================================================
REQUIREMENTS:
    pip install python-docx pandas matplotlib pillow
=============================================================
JALANKAN SETELAH:
    1. lda_modeling.py       → hasil_lda/
    2. bertopic_modeling.py  → hasil_bertopic/
    3. komparasi_lda_bertopic.py → hasil_komparasi/
=============================================================
"""

import os
import json
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ═══════════════════════════════════════════════════════════
# KONFIGURASI
# ═══════════════════════════════════════════════════════════

LDA_DIR      = "hasil_lda"
BERTOPIC_DIR = "hasil_bertopic"
KOMPARASI_DIR = "hasil_komparasi"
OUTPUT_FILE  = "Laporan_Bab4_Bab5.docx"

# Warna tema
COLOR_BIRU   = RGBColor(0x2E, 0x75, 0xB6)
COLOR_HIJAU  = RGBColor(0x28, 0xA7, 0x45)
COLOR_HEADER = RGBColor(0x2E, 0x75, 0xB6)
COLOR_SUBHEADER = RGBColor(0x1F, 0x51, 0x7F)

# ═══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════

def set_cell_bg(cell, hex_color="D5E8F0"):
    """Set background color pada cell tabel."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)


def add_heading(doc, text, level=1):
    """Tambah heading dengan styling."""
    p = doc.add_heading(text, level=level)
    run = p.runs[0] if p.runs else p.add_run(text)
    if level == 1:
        run.font.color.rgb = COLOR_BIRU
        run.font.size = Pt(14)
    elif level == 2:
        run.font.color.rgb = COLOR_SUBHEADER
        run.font.size = Pt(12)
    else:
        run.font.size = Pt(11)
    return p


def add_paragraph(doc, text, bold=False, italic=False, size=11):
    """Tambah paragraf biasa."""
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(size)
    p.paragraph_format.space_after = Pt(6)
    return p


def add_table_header_row(table, headers, bg_color="2E75B6"):
    """Tambah baris header tabel dengan background biru."""
    row = table.rows[0]
    for i, header in enumerate(headers):
        cell = row.cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        run = p.add_run(header)
        run.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.size = Pt(10)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_cell_bg(cell, bg_color)


def insert_image(doc, image_path, caption="", width_cm=14):
    """Sisipkan gambar dengan caption."""
    if not os.path.exists(image_path):
        add_paragraph(doc, f"[Gambar tidak ditemukan: {image_path}]", italic=True)
        return
    try:
        doc.add_picture(image_path, width=Cm(width_cm))
        last_para = doc.paragraphs[-1]
        last_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if caption:
            cap = doc.add_paragraph(caption)
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap.runs[0].font.size = Pt(9)
            cap.runs[0].italic = True
    except Exception as e:
        add_paragraph(doc, f"[Gagal muat gambar: {e}]", italic=True)


# ═══════════════════════════════════════════════════════════
# LOAD SEMUA DATA
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*55}")
print("  GENERATE LAPORAN BAB IV & V")
print(f"{'='*55}")

# LDA
try:
    df_lda_topics = pd.read_csv(f"{LDA_DIR}/lda_topics.csv", encoding="utf-8")
    df_lda_docs   = pd.read_csv(f"{LDA_DIR}/dokumen_dengan_topik.csv", encoding="utf-8")
    df_lda_coh    = pd.read_csv(f"{LDA_DIR}/coherence_scores.csv", encoding="utf-8")
    best_lda_k    = int(df_lda_coh.loc[df_lda_coh["coherence_cv"].idxmax(), "K"])
    best_lda_coh  = df_lda_coh["coherence_cv"].max()
    print(f"[OK] Data LDA dimuat (K={best_lda_k}, Cv={best_lda_coh:.4f})")
except Exception as e:
    print(f"[WARN] LDA data gagal dimuat: {e}")
    df_lda_topics = pd.DataFrame()
    best_lda_k, best_lda_coh = 0, 0

# BERTopic
try:
    df_bert_topics = pd.read_csv(f"{BERTOPIC_DIR}/bertopic_topics.csv", encoding="utf-8")
    df_bert_docs   = pd.read_csv(f"{BERTOPIC_DIR}/dokumen_dengan_topik_bertopic.csv", encoding="utf-8")
    with open(f"{BERTOPIC_DIR}/bertopic_metrics.json", encoding="utf-8") as f:
        bert_metrics = json.load(f)
    bert_n_topics  = bert_metrics["n_topics"]
    bert_diversity = bert_metrics["topic_diversity"]
    bert_outlier   = bert_metrics["outlier_pct"]
    print(f"[OK] Data BERTopic dimuat (K={bert_n_topics})")
except Exception as e:
    print(f"[WARN] BERTopic data gagal dimuat: {e}")
    df_bert_topics = pd.DataFrame()
    bert_n_topics, bert_diversity, bert_outlier = 0, 0, 0

# Komparasi
try:
    df_comparison = pd.read_csv(f"{KOMPARASI_DIR}/tabel_perbandingan.csv", encoding="utf-8")
    df_roadmap    = pd.read_csv(f"{KOMPARASI_DIR}/rekomendasi_roadmap.csv", encoding="utf-8")
    print(f"[OK] Data komparasi dimuat")
except Exception as e:
    print(f"[WARN] Data komparasi gagal dimuat: {e}")
    df_comparison = pd.DataFrame()
    df_roadmap    = pd.DataFrame()

# Statistik preprocessing
try:
    df_pre = pd.read_csv("data_preprocessed.csv", encoding="utf-8")
    n_total_docs = len(df_pre)
    avg_tokens   = df_pre["token_count_lda"].mean() if "token_count_lda" in df_pre.columns else 0
    year_min     = int(df_pre["year"].min()) if "year" in df_pre.columns else 2010
    year_max     = int(df_pre["year"].max()) if "year" in df_pre.columns else 2024
    print(f"[OK] Data preprocessing dimuat ({n_total_docs} dokumen)")
except Exception as e:
    print(f"[WARN] Data preprocessing gagal: {e}")
    n_total_docs, avg_tokens, year_min, year_max = 0, 0, 2010, 2024

# Diversity LDA
all_words_lda = []
if not df_lda_topics.empty:
    for ws in df_lda_topics["top_words"].dropna():
        all_words_lda.extend([w.strip() for w in ws.split(",")])
lda_diversity = round(len(set(all_words_lda)) / len(all_words_lda), 4) if all_words_lda else 0

# ═══════════════════════════════════════════════════════════
# MULAI BUAT DOKUMEN
# ═══════════════════════════════════════════════════════════

print("\nMembuat dokumen Word...")
doc = Document()

# Set margin halaman (A4, margin 2.5 cm)
section = doc.sections[0]
section.page_width  = Cm(21)
section.page_height = Cm(29.7)
for attr in ("left_margin","right_margin","top_margin","bottom_margin"):
    setattr(section, attr, Cm(2.5))

# Style default
style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(11)

# ═══════════════════════════════════════════════════════════
# BAB IV: HASIL PENELITIAN
# ═══════════════════════════════════════════════════════════

# Title
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run("BAB IV\nHASIL PENELITIAN")
run.bold = True
run.font.size = Pt(14)
run.font.color.rgb = COLOR_BIRU
doc.add_paragraph()

# ───────────────────────────────────────────────────────────
# 4.1 SELEKSI DAN AKUISISI DATA
# ───────────────────────────────────────────────────────────
add_heading(doc, "4.1 Hasil Seleksi dan Akuisisi Data", level=1)

add_paragraph(doc,
    f"Proses akuisisi data dilakukan melalui web scraping terprogram menggunakan SerpAPI "
    f"terhadap profil Google Scholar seluruh dosen sampel. Setelah melalui tahap seleksi "
    f"dan pembersihan data, diperoleh sebanyak {n_total_docs} dokumen publikasi yang valid "
    f"untuk digunakan dalam pemodelan topik, mencakup rentang tahun {year_min} hingga {year_max}."
)

add_heading(doc, "4.1.1 Statistik Dataset", level=2)

# Tabel statistik dataset
tbl = doc.add_table(rows=5, cols=2)
tbl.style = "Table Grid"
tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
headers_stat = ["Keterangan", "Nilai"]
add_table_header_row(tbl, headers_stat)

rows_stat = [
    ("Total dokumen terkumpul", str(n_total_docs)),
    (f"Rentang tahun publikasi", f"{year_min} – {year_max}"),
    ("Rata-rata panjang token (LDA)", f"{avg_tokens:.1f} token/dokumen"),
    ("Sumber data", "Google Scholar (SerpAPI)"),
]
for i, (k, v) in enumerate(rows_stat, 1):
    tbl.rows[i].cells[0].text = k
    tbl.rows[i].cells[1].text = v
    set_cell_bg(tbl.rows[i].cells[0], "EBF3FB")

doc.add_paragraph()

# ───────────────────────────────────────────────────────────
# 4.2 PRA-PEMROSESAN DAN TRANSFORMASI DATA
# ───────────────────────────────────────────────────────────
add_heading(doc, "4.2 Hasil Pra-pemrosesan dan Transformasi Data", level=1)

add_paragraph(doc,
    "Pra-pemrosesan data dilakukan secara berbeda untuk masing-masing metode sesuai "
    "kebutuhan algoritmanya. Untuk LDA, diterapkan pipeline lengkap meliputi normalisasi "
    "teks, penghapusan stopwords berbahasa Indonesia dan Inggris, tokenisasi, dan stemming "
    "menggunakan Sastrawi. Untuk BERTopic, pembersihan bersifat minimal guna mempertahankan "
    "integritas semantik teks yang dibutuhkan model transformer."
)

add_heading(doc, "4.2.1 Representasi Data Siap Model", level=2)

tbl2 = doc.add_table(rows=4, cols=3)
tbl2.style = "Table Grid"
tbl2.alignment = WD_TABLE_ALIGNMENT.CENTER
add_table_header_row(tbl2, ["Aspek", "LDA", "BERTopic"])

rows2 = [
    ("Representasi teks", "Bag-of-Words (BoW)", "Transformer Embeddings (SBERT)"),
    ("Model embedding", "—", "paraphrase-multilingual-MiniLM-L12-v2"),
    ("Preprocessing", "Stopword removal + stemming", "Minimal (hapus URL, simbol)"),
]
for i, (a, b, c) in enumerate(rows2, 1):
    tbl2.rows[i].cells[0].text = a
    tbl2.rows[i].cells[1].text = b
    tbl2.rows[i].cells[2].text = c
    set_cell_bg(tbl2.rows[i].cells[0], "EBF3FB")

doc.add_paragraph()

# ───────────────────────────────────────────────────────────
# 4.3 HASIL PEMODELAN TOPIK
# ───────────────────────────────────────────────────────────
add_heading(doc, "4.3 Hasil Pemodelan Topik", level=1)

add_heading(doc, "4.3.1 Penentuan Jumlah Topik Optimal (LDA)", level=2)
add_paragraph(doc,
    f"Jumlah topik optimal untuk LDA ditentukan melalui evaluasi nilai Topic Coherence (Cv) "
    f"pada rentang K = 2 hingga K = 15. Nilai Cv tertinggi diperoleh pada K = {best_lda_k} "
    f"dengan skor Cv = {best_lda_coh:.4f}. Grafik coherence vs. jumlah topik ditunjukkan "
    f"pada Gambar berikut."
)
insert_image(doc, f"{LDA_DIR}/coherence_vs_k.png",
             caption=f"Gambar 4.1. Grafik Topic Coherence (Cv) vs. Jumlah Topik LDA (K optimal = {best_lda_k})")

add_heading(doc, "4.3.2 Daftar Topik LDA", level=2)
add_paragraph(doc,
    f"Model LDA dengan K = {best_lda_k} menghasilkan distribusi topik berikut. "
    "Kolom kata kunci menampilkan 15 kata dengan probabilitas tertinggi per topik "
    "(label interpretasi diisi oleh peneliti)."
)

if not df_lda_topics.empty:
    n_rows = len(df_lda_topics) + 1
    tbl_lda = doc.add_table(rows=n_rows, cols=4)
    tbl_lda.style = "Table Grid"
    add_table_header_row(tbl_lda, ["ID Topik", "Label", "Kata Kunci (Top-15)", "Jml Dok"])

    # Ambil jumlah dokumen per topik
    if not df_lda_docs.empty and "dominant_topic" in df_lda_docs.columns:
        topic_counts = df_lda_docs["dominant_topic"].value_counts().to_dict()
    else:
        topic_counts = {}

    for i, row in df_lda_topics.iterrows():
        r = tbl_lda.rows[i + 1]
        tid = row.get("topic_id", i + 1)
        r.cells[0].text = str(tid)
        r.cells[1].text = str(row.get("label", f"Topik {tid}"))
        keywords = str(row.get("top_words", ""))[:120]
        r.cells[2].text = keywords
        r.cells[3].text = str(topic_counts.get(tid, "-"))
        for cell in r.cells:
            cell.paragraphs[0].runs[0].font.size = Pt(9) if cell.paragraphs[0].runs else None
        if i % 2 == 0:
            set_cell_bg(r.cells[0], "F5F9FD")
    doc.add_paragraph()

# Distribusi topik LDA
insert_image(doc, f"{LDA_DIR}/distribusi_topik.png",
             caption="Gambar 4.2. Distribusi Jumlah Dokumen per Topik LDA")

add_heading(doc, "4.3.3 Daftar Topik BERTopic", level=2)
add_paragraph(doc,
    f"BERTopic menghasilkan {bert_n_topics} topik secara otomatis melalui proses clustering "
    f"HDBSCAN pada ruang embedding multidimensional. Dokumen yang tidak masuk klaster manapun "
    f"dikategorikan sebagai outlier sebesar {bert_outlier:.1f}%."
)

if not df_bert_topics.empty:
    n_rows_b = min(len(df_bert_topics), 20) + 1  # batasi 20 topik agar tidak terlalu panjang
    tbl_bert = doc.add_table(rows=n_rows_b, cols=4)
    tbl_bert.style = "Table Grid"
    add_table_header_row(tbl_bert, ["ID Topik", "Label", "Kata Kunci (Top-15)", "Jml Dok"])

    for i, row in df_bert_topics.head(20).iterrows():
        r = tbl_bert.rows[i + 1]
        tid = row.get("topic_id", i + 1)
        r.cells[0].text = str(tid)
        r.cells[1].text = str(row.get("label", f"Topik {tid}"))
        keywords = str(row.get("top_words", ""))[:120]
        r.cells[2].text = keywords
        r.cells[3].text = str(row.get("doc_count", "-"))
        for cell in r.cells:
            if cell.paragraphs[0].runs:
                cell.paragraphs[0].runs[0].font.size = Pt(9)
        if i % 2 == 0:
            set_cell_bg(r.cells[0], "F0FBF4")
    doc.add_paragraph()

insert_image(doc, f"{BERTOPIC_DIR}/distribusi_topik_bertopic.png",
             caption="Gambar 4.3. Distribusi Jumlah Dokumen per Topik BERTopic")

# ───────────────────────────────────────────────────────────
# 4.4 EVALUASI KUANTITATIF
# ───────────────────────────────────────────────────────────
add_heading(doc, "4.4 Hasil Evaluasi Kuantitatif", level=1)

add_heading(doc, "4.4.1 Tabel Perbandingan Metrik", level=2)
add_paragraph(doc,
    "Tabel 4.1 menyajikan perbandingan komprehensif antara LDA dan BERTopic berdasarkan "
    "metrik kuantitatif maupun aspek kualitatif."
)

if not df_comparison.empty:
    n_rows_c = len(df_comparison) + 1
    tbl_cmp = doc.add_table(rows=n_rows_c, cols=3)
    tbl_cmp.style = "Table Grid"
    tbl_cmp.alignment = WD_TABLE_ALIGNMENT.CENTER
    add_table_header_row(tbl_cmp, ["Metrik", "LDA", "BERTopic"])

    for i, row in df_comparison.iterrows():
        r = tbl_cmp.rows[i + 1]
        r.cells[0].text = str(row.get("Metrik", ""))
        r.cells[1].text = str(row.get("LDA", ""))
        r.cells[2].text = str(row.get("BERTopic", ""))
        set_cell_bg(r.cells[0], "EBF3FB")
        if i % 2 == 0:
            set_cell_bg(r.cells[1], "F5F9FD")
            set_cell_bg(r.cells[2], "F0FBF4")
    doc.add_paragraph()

cap_cmp = doc.add_paragraph("Tabel 4.1. Perbandingan Metrik LDA vs BERTopic")
cap_cmp.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap_cmp.runs[0].font.size = Pt(9)
cap_cmp.runs[0].italic = True

insert_image(doc, f"{KOMPARASI_DIR}/komparasi_metrik.png",
             caption="Gambar 4.4. Visualisasi Perbandingan Topic Coherence, Diversity, dan Jumlah Topik")

add_heading(doc, "4.4.2 Analisis Temporal", level=2)
add_paragraph(doc,
    "Analisis temporal dilakukan untuk menelusuri dinamika perkembangan topik riset "
    "dari tahun ke tahun. Grafik berikut menunjukkan proporsi dokumen per topik "
    "utama untuk masing-masing metode."
)
insert_image(doc, f"{LDA_DIR}/tren_topik_per_tahun.png",
             caption="Gambar 4.5. Tren Topik LDA per Tahun (5 Topik Terbesar)")
insert_image(doc, f"{BERTOPIC_DIR}/tren_topik_bertopic_per_tahun.png",
             caption="Gambar 4.6. Tren Topik BERTopic per Tahun (5 Topik Terbesar)")
insert_image(doc, f"{KOMPARASI_DIR}/tren_temporal_overlay.png",
             caption="Gambar 4.7. Overlay Tren Temporal LDA vs BERTopic")

# ═══════════════════════════════════════════════════════════
# BAB V: PEMBAHASAN
# ═══════════════════════════════════════════════════════════

doc.add_page_break()

title2 = doc.add_paragraph()
title2.alignment = WD_ALIGN_PARAGRAPH.CENTER
run2 = title2.add_run("BAB V\nPEMBAHASAN PENELITIAN")
run2.bold = True
run2.font.size = Pt(14)
run2.font.color.rgb = COLOR_BIRU
doc.add_paragraph()

# ───────────────────────────────────────────────────────────
# 5.1 KARAKTERISTIK DAN DISTRIBUSI TOPIK
# ───────────────────────────────────────────────────────────
add_heading(doc, "5.1 Pembahasan Karakteristik dan Distribusi Topik Riset", level=1)
add_paragraph(doc,
    f"Pemodelan menggunakan LDA menghasilkan {best_lda_k} topik utama yang secara statistik "
    f"menangkap struktur semantik laten dari korpus publikasi dosen. Sementara itu, BERTopic "
    f"secara adaptif menemukan {bert_n_topics} topik melalui clustering berbasis embedding "
    f"kontekstual. Kedua metode menunjukkan konsentrasi topik yang berbeda: LDA cenderung "
    f"menghasilkan topik yang lebih luas (broad), sedangkan BERTopic menghasilkan klaster "
    f"yang lebih spesifik dan kontekstual."
)

add_heading(doc, "5.1.1 Life Cycle Topik LDA", level=2)
add_paragraph(doc,
    "Klasifikasi topik berdasarkan siklus hidup (life cycle) menunjukkan distribusi "
    "fase emerging, growth, maturity, dan declining sebagaimana ditunjukkan pada "
    "Gambar 5.1 berikut."
)
insert_image(doc, f"{KOMPARASI_DIR}/lifecycle_bubble_lda.png",
             caption="Gambar 5.1. Life Cycle Topik LDA (ukuran bubble = jumlah dokumen)")

add_heading(doc, "5.1.2 Life Cycle Topik BERTopic", level=2)
insert_image(doc, f"{KOMPARASI_DIR}/lifecycle_bubble_bertopic.png",
             caption="Gambar 5.2. Life Cycle Topik BERTopic (ukuran bubble = jumlah dokumen)")

# ───────────────────────────────────────────────────────────
# 5.2 PERBANDINGAN KINERJA
# ───────────────────────────────────────────────────────────
add_heading(doc, "5.2 Pembahasan Perbandingan Kinerja LDA dan BERTopic", level=1)

add_paragraph(doc,
    f"Berdasarkan hasil evaluasi kuantitatif, LDA menghasilkan nilai Topic Coherence "
    f"(Cv) sebesar {best_lda_coh:.4f} dengan Topic Diversity sebesar {lda_diversity:.4f}. "
    f"BERTopic menghasilkan {bert_n_topics} topik dengan Topic Diversity sebesar "
    f"{bert_diversity:.4f} dan tingkat outlier {bert_outlier:.1f}%."
)

add_paragraph(doc,
    "Kelebihan BERTopic terletak pada kemampuan model transformer dalam menangkap konteks "
    "semantik, yang menghasilkan topik lebih koheren pada teks pendek (judul publikasi). "
    "Sementara LDA unggul dalam kecepatan komputasi, kemudahan interpretasi secara statistik, "
    "dan tidak menghasilkan dokumen outlier sehingga semua data terklasifikasi."
)

# ───────────────────────────────────────────────────────────
# 5.3 ROADMAP PENELITIAN STRATEGIS
# ───────────────────────────────────────────────────────────
add_heading(doc, "5.3 Pembahasan Integrasi ke dalam Roadmap Penelitian Strategis", level=1)

add_paragraph(doc,
    "Berdasarkan analisis life cycle topik dari kedua metode, disusun rekomendasi roadmap "
    "penelitian strategis yang memetakan topik ke dalam tiga fase utama: Emerging "
    "(perlu dukungan hibah awal), Growth (prioritas kolaborasi dan publikasi internasional), "
    "dan Maturity (konsolidasi dan integrasi kurikulum)."
)

add_heading(doc, "5.3.1 Rekomendasi Roadmap (Gabungan LDA + BERTopic)", level=2)

if not df_roadmap.empty:
    phase_order = ["Emerging", "Growth", "Maturity", "Declining"]
    for phase in phase_order:
        subset = df_roadmap[df_roadmap["fase"] == phase]
        if subset.empty:
            continue

        phase_colors_hex = {
            "Emerging" : "FFF3E0",
            "Growth"   : "E8F5E9",
            "Maturity" : "E3F2FD",
            "Declining": "F5F5F5",
        }
        add_heading(doc, f"Fase {phase}", level=3)

        n_r = len(subset) + 1
        tbl_r = doc.add_table(rows=n_r, cols=4)
        tbl_r.style = "Table Grid"
        add_table_header_row(tbl_r, ["Model", "ID Topik", "Kata Kunci", "Rekomendasi"])

        for i, (_, row) in enumerate(subset.iterrows(), 1):
            r = tbl_r.rows[i]
            r.cells[0].text = str(row.get("model", ""))
            r.cells[1].text = str(row.get("topik_id", ""))
            r.cells[2].text = str(row.get("kata_kunci", ""))[:80]
            r.cells[3].text = str(row.get("rekomendasi", ""))
            for cell in r.cells:
                if cell.paragraphs[0].runs:
                    cell.paragraphs[0].runs[0].font.size = Pt(9)
            set_cell_bg(r.cells[0], phase_colors_hex.get(phase, "FFFFFF"))
        doc.add_paragraph()

# ───────────────────────────────────────────────────────────
# 5.4 KETERBATASAN DAN SARAN
# ───────────────────────────────────────────────────────────
add_heading(doc, "5.4 Diskusi Keterbatasan dan Saran", level=1)

add_paragraph(doc,
    "Beberapa keterbatasan penelitian ini perlu diakui. Pertama, sumber data yang digunakan "
    "terbatas pada Google Scholar sehingga publikasi yang tidak terindeks di platform tersebut "
    "tidak tercakup. Kedua, analisis dilakukan pada data judul publikasi tanpa abstrak penuh, "
    "yang dapat mempengaruhi kekayaan semantik representasi dokumen. Ketiga, pelabelan topik "
    "masih bersifat interpretatif dan memerlukan validasi dari pakar domain."
)

add_paragraph(doc,
    "Untuk penelitian lanjutan, disarankan: (1) menggunakan model embedding bahasa Indonesia "
    "yang lebih spesifik seperti IndoBERT; (2) mengintegrasikan data abstrak dan sitasi; "
    "(3) memperluas cakupan institusi untuk generalisasi hasil yang lebih luas; dan "
    "(4) melakukan validasi kualitatif dengan wawancara dosen atau kepala program studi."
)

# ═══════════════════════════════════════════════════════════
# SIMPAN DOKUMEN
# ═══════════════════════════════════════════════════════════

doc.save(OUTPUT_FILE)
print(f"\n{'='*55}")
print(f"  ✓ Laporan berhasil dibuat: {OUTPUT_FILE}")
print(f"{'='*55}")
print(f"\n  Isi dokumen:")
print(f"  - Bab IV (4.1 – 4.4): Hasil Penelitian + tabel + grafik")
print(f"  - Bab V  (5.1 – 5.4): Pembahasan + roadmap + keterbatasan")
print(f"\n  LANGKAH SELANJUTNYA:")
print(f"  1. Buka {OUTPUT_FILE} di Word")
print(f"  2. Isi kolom 'Label' pada tabel topik (beri nama tiap topik)")
print(f"  3. Salin-tempel Bab IV & V ke dalam HIbah2026.docx")
print(f"  4. Sesuaikan narasi dengan temuan aktual data kamu")
print(f"{'='*55}\n")
