# Yelp and YellowPages Sentiment Analysis

This repo contains an end to end sentiment analysis pipeline for healthcare location reviews from Yelp and YellowPages.

## What it does

1. Loads `totaldata.csv` from the `data` folder. The CSV must have these columns: `rating, review, location`.
2. Cleans text and builds simple text features.
3. Runs sentiment:
   - Uses a Transformer model (DistilBERT sentiment) when available.
   - Falls back to TextBlob if the model cannot be downloaded in your environment.
4. Produces overall and per location summaries.
5. Ranks locations by positive percent and by average sentiment score.
6. Exports CSVs and a few quick charts into the `output` folder.
