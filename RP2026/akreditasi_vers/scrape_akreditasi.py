import requests
import json
import csv
import time

# =============================================
# GANTI DENGAN API KEY SERPAPI KAMU
API_KEY = "07f86771c2dd11108ca4b637519eddd4f77f20aff08bd354b95b368be49fbf95"
# =============================================

# Daftar semua Author ID dari link yang diberikan
AUTHORS = [
    "mPIjC2oAAAAJ",
    "dV2vWw8AAAAJ",
    "prEYUoUAAAAJ",
    "XWyNzrsAAAAJ",
    "0Z9tTNMAAAAJ",
    "EMt4MqkAAAAJ",
    "U-NI-xYAAAAJ",
    "4jrEgZsAAAAJ",
    "wzWX4oQAAAAJ",
    "Og1jkq8AAAAJ",
    "W9uYTXUAAAAJ",
    "-mYDEosAAAAJ",
    "vIKtPwoAAAAJ",
    "sRs85fMAAAAJ",
    "2uwxYVMAAAAJ",
    "ExCZ-CcAAAAJ",
    "IwGaBnUAAAAJ",
    "3upoKAIAAAAJ",
    "vTU6xmgAAAAJ",
    "QzUL5X0AAAAJ",
    "r0m7xlMAAAAJ",
    "S_8Nz2sAAAAJ",   # Pak Aang
    "R1txw1gAAAAJ",   # Pak Qomarul
    "rilu7HgAAAAJ",   # Pak Husni Teja
]


def get_author_data(author_id):
    """Ambil data profil + artikel dari satu author"""
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
    """Ambil info penting dari response JSON"""
    author = data.get("author", {})
    cited_by = data.get("cited_by", {})

    # Ambil statistik sitasi dari tabel
    stats = {}
    for row in cited_by.get("table", []):
        for key, val in row.items():
            stats[key] = val.get("all", "-")

    # Ambil daftar artikel
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


# =============================================
# MULAI SCRAPING
# =============================================
all_authors_data = []
all_articles_data = []

print(f"Mulai scraping {len(AUTHORS)} author...\n")

for i, author_id in enumerate(AUTHORS, 1):
    print(f"[{i}/{len(AUTHORS)}] Scraping author ID: {author_id} ...", end=" ")
    try:
        raw = get_author_data(author_id)

        # Cek apakah ada error dari API
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

        # Simpan semua artikel dengan info author sekaligus
        for article in info["articles"]:
            all_articles_data.append({
                "author_id": author_id,
                "author_name": info["name"],
                "affiliations": info["affiliations"],
                "email": info["email"],
                "website": info["website"],
                "total_citations": info["total_citations"],
                "h_index": info["h_index"],
                "i10_index": info["i10_index"],
                **article,
            })

        print(f"OK - {info['name']} ({len(info['articles'])} artikel)")

    except Exception as e:
        print(f"GAGAL - {e}")

    # Jeda 1 detik antar request agar tidak kena rate limit
    time.sleep(1)

# =============================================
# SIMPAN HASIL
# =============================================

with open("hasil_scraping.csv", "w", newline="", encoding="utf-8") as f:
    fieldnames = ["author_id", "author_name", "affiliations", "email", "website",
                  "total_citations", "h_index", "i10_index",
                  "title", "authors", "publication", "year", "cited_by"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_articles_data)

print("\n✅ Semua data disimpan: hasil_scraping.csv")
print(f"\nSelesai! Total {len(all_authors_data)} author, {len(all_articles_data)} artikel.")
