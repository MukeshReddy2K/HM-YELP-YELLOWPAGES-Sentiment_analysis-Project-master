import os
import re
import ssl
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd

# NLP
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob

# ML and viz
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

TRANSFORMERS_OK = True
try:
    from transformers import pipeline as hf_pipeline
except Exception:
    TRANSFORMERS_OK = False

try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except Exception:
    pass

for pkg in ["punkt", "stopwords"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

BASE_PATH = Path(__file__).parent if "__file__" in globals() else Path.cwd()
DATA_PATH = BASE_PATH / "Data"
OUTPUT_PATH = BASE_PATH / "OutPuts"
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|[\w\.-]+@[\w\.-]+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data() -> pd.DataFrame:
    fp = DATA_PATH / "totaldata.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Could not find {fp}. Place totaldata.csv under the data folder.")
    df = pd.read_csv(fp)
    df.columns = [c.lower().strip() for c in df.columns]
    expected = {"rating", "review", "location"}
    if not expected.issubset(set(df.columns)):
        raise ValueError(f"CSV must have columns: rating, review, location. Found: {list(df.columns)}")
    return df


def try_load_transformer() -> Optional[object]:
    if not TRANSFORMERS_OK:
        return None
    try:
        clf = hf_pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        return clf
    except Exception:
        return None


def score_sentiment_transformer(texts, clf) -> Tuple[np.ndarray, np.ndarray]:
    labels = []
    scores = []
    for t in texts:
        if not t or not isinstance(t, str):
            labels.append("NEUTRAL")
            scores.append(0.0)
            continue
        res = clf(t[:512])[0]
        labels.append(res.get("label", "NEUTRAL").upper())
        scores.append(float(res.get("score", 0.0)))
    return np.array(labels), np.array(scores, dtype=float)


def score_sentiment_textblob(texts) -> Tuple[np.ndarray, np.ndarray]:
    labels = []
    scores = []
    for t in texts:
        if not t or not isinstance(t, str):
            labels.append("NEUTRAL")
            scores.append(0.0)
            continue
        pol = TextBlob(t).sentiment.polarity
        if pol > 0.05:
            labels.append("POSITIVE")
            scores.append(float(pol))
        elif pol < -0.05:
            labels.append("NEGATIVE")
            scores.append(float(abs(pol)))
        else:
            labels.append("NEUTRAL")
            scores.append(0.0)
    return np.array(labels), np.array(scores, dtype=float)


def add_basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    df["review"] = df["review"].astype(str)
    df["cleaned_text"] = df["review"].apply(clean_text)
    df["char_len"] = df["review"].str.len()
    df["word_len"] = df["cleaned_text"].apply(lambda x: len(x.split()))
    return df


def aspect_ngrams(df: pd.DataFrame, label_filter: Optional[str] = None, top_k: int = 20) -> pd.DataFrame:
    sub = df if label_filter is None else df[df["sentiment_label"] == label_filter]
    corpus = sub["cleaned_text"].tolist()
    if len(corpus) == 0:
        return pd.DataFrame(columns=["term", "score"])

    sw = set(stopwords.words("english"))
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words=sw)
    try:
        X = vec.fit_transform(corpus)
    except ValueError:
        return pd.DataFrame(columns=["term", "score"])
    scores = np.asarray(X.sum(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())
    top_idx = scores.argsort()[::-1][:top_k]
    return pd.DataFrame({"term": terms[top_idx], "score": scores[top_idx]})


def plot_distribution(series: pd.Series, title: str, fname: str):
    ax = series.value_counts().plot(kind="bar", title=title)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / fname)
    plt.close()


def run():
    df = load_data()
    df = add_basic_stats(df)

    clf = try_load_transformer()
    if clf is not None:
        labels, scores = score_sentiment_transformer(df["cleaned_text"].tolist(), clf)
        engine = "transformers"
    else:
        labels, scores = score_sentiment_textblob(df["cleaned_text"].tolist())
        engine = "textblob"

    df["sentiment_label"] = labels
    df["sentiment_score"] = scores

    overall_dist = df["sentiment_label"].value_counts(normalize=True).rename_axis("sentiment").reset_index(name="pct")
    overall_dist["pct"] = (overall_dist["pct"] * 100).round(2)

    by_loc = (
        df.groupby("location")
          .agg(
              n_reviews=("review", "count"),
              avg_rating=("rating", "mean"),
              pos_pct=("sentiment_label", lambda s: (s == "POSITIVE").mean() * 100),
              neg_pct=("sentiment_label", lambda s: (s == "NEGATIVE").mean() * 100),
              neu_pct=("sentiment_label", lambda s: (s == "NEUTRAL").mean() * 100),
              avg_sentiment_score=("sentiment_score", "mean"),
              median_words=("word_len", "median")
          )
          .reset_index()
    )
    rank_by_pos = by_loc.sort_values(["pos_pct", "avg_rating", "n_reviews"], ascending=[False, False, False]).reset_index(drop=True)
    rank_by_score = by_loc.sort_values(["avg_sentiment_score", "avg_rating", "n_reviews"], ascending=[False, False, False]).reset_index(drop=True)

    df["rating_num"] = pd.to_numeric(df["rating"], errors="coerce")
    corr = df[["rating_num", "sentiment_score", "word_len", "char_len"]].corr(numeric_only=True)

    top_all = aspect_ngrams(df, None, top_k=25)
    top_pos = aspect_ngrams(df, "POSITIVE", top_k=15)
    top_neg = aspect_ngrams(df, "NEGATIVE", top_k=15)

    df.to_csv(OUTPUT_PATH / "reviews_with_sentiment.csv", index=False)
    overall_dist.to_csv(OUTPUT_PATH / "overall_sentiment_distribution.csv", index=False)
    by_loc.to_csv(OUTPUT_PATH / "by_location_summary.csv", index=False)
    rank_by_pos.to_csv(OUTPUT_PATH / "ranking_by_positive_pct.csv", index=False)
    rank_by_score.to_csv(OUTPUT_PATH / "ranking_by_sentiment_score.csv", index=False)
    corr.to_csv(OUTPUT_PATH / "correlations.csv")

    plot_distribution(df["sentiment_label"], "Overall Sentiment Distribution", "plot_overall_sentiment.png")
    plot_distribution(df["location"], "Review count by location", "plot_reviews_by_location.png")

    print("\n=== Quick Report ===")
    print(f"Engine used: {engine}")
    if not overall_dist.empty:
        print("\nOverall sentiment distribution percent:")
        print(overall_dist.to_string(index=False))
    print("\nTop 5 locations by positive percent:")
    print(rank_by_pos[["location", "n_reviews", "avg_rating", "pos_pct"]].head(5).to_string(index=False))
    print("\nTop 5 locations by avg sentiment score:")
    print(rank_by_score[["location", "n_reviews", "avg_rating", "avg_sentiment_score"]].head(5).to_string(index=False))
    print("\nCorrelation matrix (rating vs sentiment and lengths):")
    print(corr.to_string())

    print(f"\nOutputs saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    run()

