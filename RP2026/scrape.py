import requests
import json
import csv
import time
import os
from difflib import SequenceMatcher
from dotenv import load_dotenv
from deep_translator import GoogleTranslator

# Muat isi file .env ke sistem
load_dotenv()

# =============================================
# KONFIGURASI API KEY & FILE CACHE
# =============================================
API_KEY = os.getenv("SERPAPI_KEY")
if not API_KEY:
    raise ValueError("API Key SerpAPI tidak ditemukan! Pastikan file .env sudah dikonfigurasi.")

CACHE_FILE = "translation_cache.json"
CACHE_SAVE_EVERY = 20          # simpan cache tiap N judul, bukan tiap 1 judul
MIN_TITLE_SIMILARITY = 0.85    # ambang kemiripan judul agar abstrak Semantic Scholar diterima
SLEEP_BETWEEN_TITLES = 1.5     # detik, jeda antar judul (Semantic Scholar + Google Translate)

# Daftar semua Author ID
AUTHORS = [
    "mPIjC2oAAAAJ", "dV2vWw8AAAAJ", "prEYUoUAAAAJ", "XWyNzrsAAAAJ",
    "0Z9tTNMAAAAJ", "EMt4MqkAAAAJ", "U-NI-xYAAAAJ", "4jrEgZsAAAAJ",
    "wzWX4oQAAAAJ", "Og1jkq8AAAAJ", "W9uYTXUAAAAJ", "-mYDEosAAAAJ",
    "vIKtPwoAAAAJ", "sRs85fMAAAAJ", "2uwxYVMAAAAJ", "ExCZ-CcAAAAJ",
    "IwGaBnUAAAAJ", "3upoKAIAAAAJ", "vTU6xmgAAAAJ", "QzUL5X0AAAAJ",
    "r0m7xlMAAAAJ"
]

# =============================================
# FUNGSI-FUNGSI UTAMA
# =============================================

def load_cache():
    """Memuat cache lokal jika ada"""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache_data):
    """Menyimpan cache lokal agar aman jika script terhenti"""
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)


def title_similarity(a: str, b: str) -> float:
    """Rasio kemiripan string 0-1, dipakai untuk validasi hasil pencarian
    Semantic Scholar sebelum abstraknya diterima."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def get_author_data(author_id):
    """Ambil data profil + artikel dari satu author via SerpAPI"""
    url = "https://serpapi.com/search"
    params = {
        "engine": "google_scholar_author",
        "author_id": author_id,
        "hl": "en",
        "num": 100,
        "api_key": API_KEY,
    }
    response = requests.get(url, params=params)
    return response.json()


def extract_info(data):
    """Ambil info penting dari response JSON SerpAPI"""
    author = data.get("author", {})
    cited_by = data.get("cited_by", {})

    stats = {}
    for row in cited_by.get("table", []):
        for key, val in row.items():
            stats[key] = val.get("all", "-")

    articles = []
    for a in data.get("articles", []):
        articles.append({
            "title": a.get("title", "-"),
            "authors": a.get("authors", "-"),
            "publication": a.get("publication", "-"),
            "year": a.get("year", "-"),
            "cited_by": a.get("cited_by", {}).get("value", 0),
        })

    return {
        "name": author.get("name", "-"),
        "affiliations": author.get("affiliations", "-"),
        "email": author.get("email", "-"),
        "website": author.get("website", "-"),
        "total_citations": stats.get("citations", "-"),
        "h_index": stats.get("h_index", "-"),
        "i10_index": stats.get("i10_index", "-"),
        "articles": articles,
    }


def fetch_abstract_and_translate(title, translator, cache):
    """
    Cari abstrak di Semantic Scholar (DENGAN validasi kemiripan judul agar
    tidak salah ambil abstrak dari paper yang berbeda), lalu terjemahkan
    judul & abstrak ke Bahasa Inggris.

    NOTE: fungsi ini TIDAK memanggil save_cache() sendiri — pemanggilnya
    (loop di bawah) yang bertanggung jawab menyimpan cache secara berkala,
    supaya tidak menulis ulang file cache di setiap 1 judul.
    """
    if title in cache:
        return cache[title]

    abstract = ""
    title_en = title
    abstract_en = ""

    # 1. Tarik Abstrak dari Semantic Scholar (Gratis) — validasi kemiripan
    #    judul dulu sebelum abstraknya diterima.
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {"query": title, "limit": 3, "fields": "title,abstract"}
        res = requests.get(url, params=params, timeout=10)

        if res.status_code == 200:
            candidates = res.json().get("data", [])
            best, best_sim = None, 0.0
            for cand in candidates:
                sim = title_similarity(title, cand.get("title", "") or "")
                if sim > best_sim:
                    best, best_sim = cand, sim
            if best is not None and best_sim >= MIN_TITLE_SIMILARITY:
                abstract = best.get("abstract") or ""
        elif res.status_code == 429:
            # Rate limited — beda kasus dari "memang tidak ketemu abstrak"
            print(f"\n[Warning] Rate limited Semantic Scholar untuk '{title[:30]}...' — dilewati")
    except Exception as e:
        print(f"\n[Warning] Gagal mengambil abstrak untuk '{title[:30]}...': {e}")

    # 2. Terjemahkan ke Bahasa Inggris
    try:
        title_en = translator.translate(title)
        if abstract:
            # Batasi panjang teks agar tidak membebani translator gratis (~4000 karakter max)
            abstract_en = translator.translate(abstract[:4000])
    except Exception as e:
        print(f"\n[Warning] Gagal menerjemahkan '{title[:30]}...': {e}")
        title_en = title
        abstract_en = abstract

    # 3. Gabungkan Judul & Abstrak untuk Bahan Modeling
    combined_text_en = f"{title_en}. {abstract_en}".strip() if abstract_en else title_en

    result = {
        "title_en": title_en,
        "abstract": abstract,
        "abstract_en": abstract_en,
        "combined_text_en": combined_text_en,
        "has_abstract": bool(abstract_en),
    }

    cache[title] = result
    return result


# =============================================
# JALANKAN PROSES UTAMA
# =============================================

translator = GoogleTranslator(source='auto', target='en')
cache = load_cache()

all_authors_data = []
all_articles_data = []

print(f"==================================================")
print(f"Langkah 1: Scraping Data Author & Artikel via SerpAPI")
print(f"==================================================")

for i, author_id in enumerate(AUTHORS, 1):
    print(f"[{i}/{len(AUTHORS)}] Scraping author ID: {author_id} ...", end=" ")
    try:
        raw = get_author_data(author_id)

        if "error" in raw:
            print(f"ERROR: {raw['error']}")
            continue

        info = extract_info(raw)
        all_authors_data.append({
            "author_id": author_id,
            "name": info["name"],
            "affiliations": info["affiliations"],
            "email": info["email"],
            "website": info["website"],
            "total_citations": info["total_citations"],
            "h_index": info["h_index"],
            "i10_index": info["i10_index"],
        })

        for article in info["articles"]:
            all_articles_data.append({
                "author_id": author_id,
                "author_name": info["name"],
                **article,
            })

        print(f"OK - {info['name']} ({len(info['articles'])} artikel)")

    except Exception as e:
        print(f"GAGAL - {e}")

    time.sleep(1)  # Jeda aman SerpAPI

# Simpan ringkasan author
with open("ringkasan_author.csv", "w", newline="", encoding="utf-8") as f:
    fieldnames = ["author_id", "name", "affiliations", "email", "website",
                  "total_citations", "h_index", "i10_index"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_authors_data)

print(f"\n✅ Langkah 1 Selesai! Ringkasan author disimpan.")

# ---------------------------------------------
# LANGKAH 2: TARIK ABSTRAK & TERJEMAHKAN (DENGAN DEDUPLIKASI)
# ---------------------------------------------
print(f"\n==================================================")
print(f"Langkah 2: Ambil Abstrak & Terjemahkan ke Inggris")
print(f"==================================================")

# Cari judul unik agar tidak melakukan request double untuk artikel yang sama
unique_titles = list(set([a["title"] for a in all_articles_data if a["title"] != "-"]))
total_unique = len(unique_titles)

print(f"Menemukan total {len(all_articles_data)} artikel (dengan {total_unique} judul unik).")
print(f"Memproses penarikan abstrak & translasi...")

n_with_abstract = 0
for idx, title in enumerate(unique_titles, 1):
    print(f"\rMemproses artikel [{idx}/{total_unique}]: {title[:40]}...", end="", flush=True)

    result = fetch_abstract_and_translate(title, translator, cache)
    if result.get("has_abstract"):
        n_with_abstract += 1

    # Simpan cache secara berkala, bukan tiap 1 judul (lebih cepat)
    if idx % CACHE_SAVE_EVERY == 0:
        save_cache(cache)

    time.sleep(SLEEP_BETWEEN_TITLES)

save_cache(cache)  # simpan sekali lagi di akhir, jaga-jaga

print("\n\n✅ Selesai memproses abstrak dan penerjemahan!")
print(f"   Judul unik dapat abstrak: {n_with_abstract}/{total_unique} "
      f"({n_with_abstract/total_unique*100:.1f}%)")

# Map kembali hasil translasi ke seluruh artikel yang sudah di-scrape
for article in all_articles_data:
    title = article["title"]
    if title in cache:
        article["title_en"] = cache[title]["title_en"]
        article["abstract_en"] = cache[title]["abstract_en"]
        article["combined_text_en"] = cache[title]["combined_text_en"]
        article["has_abstract"] = cache[title].get("has_abstract", bool(cache[title]["abstract_en"]))
    else:
        article["title_en"] = title
        article["abstract_en"] = ""
        article["combined_text_en"] = title
        article["has_abstract"] = False

# Simpan semua artikel yang sudah diperkaya ke CSV
with open("semua_artikel.csv", "w", newline="", encoding="utf-8") as f:
    fieldnames = ["author_id", "author_name", "title", "title_en", "abstract_en",
                  "combined_text_en", "has_abstract", "authors", "publication",
                  "year", "cited_by"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_articles_data)

print("✅ Semua artikel kaya data disimpan di: semua_artikel.csv")
print(f"Selesai! Total {len(all_authors_data)} author dan {len(all_articles_data)} artikel siap diolah.")