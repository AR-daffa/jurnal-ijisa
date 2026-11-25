"""
=============================================================
FAIR COHERENCE
Menghitung Topic Coherence (Cv) LDA vs BERTopic di atas korpus
REFERENSI YANG SAMA, supaya perbandingan head-to-head valid.

MASALAH YANG DISELESAIKAN:
    1) Sebelumnya, coherence LDA dihitung dari teks yang sudah di-stem +
       stopword-removed (title_clean_lda), sedangkan coherence BERTopic
       dihitung dari teks minimal-cleaning (title_clean_bertopic).
       Preprocessing yang beda ini ikut memengaruhi skor Cv, di luar
       kualitas model itu sendiri.

    2) [FIX BARU] Bahkan setelah dipakai satu korpus referensi yang sama
       (title_clean_bertopic), tetap ada bias tersembunyi: kata-kata topik
       LDA berbentuk LEMMA (mis. "network", "system") karena
       preprocess_for_lda() melakukan lemmatization, sedangkan korpus
       referensi (title_clean_bertopic) TIDAK dilemmatize — masih ada
       "networks", "systems", dst. Akibatnya token co-occurrence untuk
       LDA gagal match ke bentuk kata infleksi di korpus referensi, dan
       skor coherence LDA jadi under-estimated bukan karena kualitas
       topiknya, tapi karena mismatch bentuk kata.

SOLUSI:
    Baik korpus referensi maupun kata-kata topik dari KEDUA model
    di-lemmatize dengan lemmatizer yang SAMA (dari preprocessing.py)
    sebelum dictionary dibangun dan sebelum coherence dihitung. Dengan
    begitu, LDA dan BERTopic dibandingkan di ruang kata yang sudah
    dinormalisasi secara konsisten.

CARA PAKAI (di komparasi_lda_bertopic.py) — TIDAK BERUBAH:

    from fair_coherence import compute_fair_coherence

    result = compute_fair_coherence(
        lda_topic_words=lda_topic_words,       # list[list[str]]
        bert_topic_words=bert_topic_words,     # list[list[str]]
        reference_texts=reference_texts,       # list[str], SATU sumber untuk keduanya
    )
    print(result["lda_coherence_fair"], result["bert_coherence_fair"])
=============================================================
"""

from gensim import corpora
from gensim.models import CoherenceModel

# Pakai lemmatizer YANG SAMA dengan preprocessing.py (preprocess_for_lda),
# supaya bentuk lemma di sini konsisten dengan bentuk lemma kata topik LDA.
from preprocessing import lemmatizer


def _lemmatize_word(w: str) -> str:
    return lemmatizer.lemmatize(w.lower().strip())


def _tokenize_and_lemmatize(texts):
    """Tokenisasi teks referensi + lemmatize tiap token, supaya korpus
    referensi berada di ruang kata yang sama dengan kata-kata topik
    (yang juga di-lemmatize di bawah)."""
    tokenized = []
    for t in texts:
        if not isinstance(t, str) or len(t.strip()) == 0:
            continue
        tokens = [_lemmatize_word(w) for w in t.split() if w.strip()]
        tokenized.append(tokens)
    return tokenized


# GANTI DENGAN KODE BARU INI:
def _lemmatize_topic_words(topic_words_list):
    """Lemmatize kata-kata topik (dipakai untuk LDA & BERTopic dua-duanya).
    [FIX] BERTopic memakai ngram_range=(1, 2), jadi sebagian top_words-nya
    adalah frasa dua-kata (mis. "knowledge management", "user satisfaction").
    Kalau frasa ini di-lemmatize dan dicocokkan ke dictionary sebagai SATU
    string utuh, ia tidak akan pernah ketemu (dictionary referensi hanya
    berisi kata tunggal dari title_clean_bertopic.split()) — sehingga
    keyword dua-kata BERTopic diam-diam terbuang dari perhitungan coherence,
    secara tidak merata antar topik (Topic 4 misalnya kehilangan 5 dari 10
    kata kuncinya). Perbaikannya: pecah tiap frasa jadi kata-kata penyusun,
    baru dilemmatize per kata, supaya konsisten dengan granularitas token
    di dictionary referensi (dan supaya LDA & BERTopic dievaluasi di level
    yang sama: kata tunggal).
    """
    out = []
    for topic in topic_words_list:
        topic_unigrams = []
        seen = set()
        for phrase in topic:
            for w in str(phrase).split():
                lw = _lemmatize_word(w)
                if lw and lw not in seen:
                    seen.add(lw)
                    topic_unigrams.append(lw)
        out.append(topic_unigrams)
    return out


def compute_fair_coherence(lda_topic_words, bert_topic_words, reference_texts):
    """
    lda_topic_words / bert_topic_words : list of list of str
        Contoh: [["system","design","web"], ["mobile","user","app"], ...]
        (ambil dari kolom top_words masing-masing model, top-N kata per topik)
    reference_texts : list of str
        Korpus referensi TUNGGAL yang dipakai untuk MENGHITUNG coherence
        kedua model. Disarankan pakai title_clean_bertopic dari
        data_preprocessed.csv (dataset lengkap, bukan hasil filter
        masing-masing model), karena itu representasi teks paling minim
        distorsi yang tersedia untuk keduanya.
    """
    tokenized_ref = _tokenize_and_lemmatize(reference_texts)
    if not tokenized_ref:
        raise ValueError("reference_texts kosong setelah tokenisasi — cek datanya.")

    dictionary = corpora.Dictionary(tokenized_ref)

    # Lemmatize kata topik SEBELUM dicocokkan ke dictionary — ini bagian
    # yang memperbaiki bias: dulu hanya reference_texts yang diproses,
    # sekarang topic words-nya juga disamakan bentuknya.
    lda_topic_words_lemma  = _lemmatize_topic_words(lda_topic_words)
    bert_topic_words_lemma = _lemmatize_topic_words(bert_topic_words)

    def _score(topic_words):
        # Buang kata yang tidak ada sama sekali di dictionary referensi,
        # gensim akan error kalau ada out-of-vocabulary di semua topik.
        cleaned = [[w for w in topic if w in dictionary.token2id] for topic in topic_words]
        cleaned = [t for t in cleaned if len(t) > 0]
        if not cleaned:
            return None
        cm = CoherenceModel(
            topics=cleaned,
            texts=tokenized_ref,
            dictionary=dictionary,
            coherence="c_v",
            processes=1,
        )
        return cm.get_coherence()

    lda_score = _score(lda_topic_words_lemma)
    bert_score = _score(bert_topic_words_lemma)

    return {
        "lda_coherence_fair": lda_score,
        "bert_coherence_fair": bert_score,
        "n_reference_docs": len(tokenized_ref),
        "note": (
            "Kedua skor dihitung dari korpus referensi yang sama "
            "(title_clean_bertopic dari data_preprocessed.csv), dan "
            "kata-kata topik kedua model dilemmatize dengan lemmatizer "
            "yang sama sebelum dicocokkan ke korpus referensi — supaya "
            "tidak ada bias akibat mismatch bentuk kata (lemma vs "
            "infleksi). Berbeda dari coherence 'native' masing-masing "
            "model yang dihitung dari korpus training-nya sendiri."
        ),
    }