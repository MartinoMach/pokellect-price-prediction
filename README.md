# Predicting Trading Card Prices via Supervised Learning

This project applies **Regression-based Supervised Learning** to predict the market price of trading cards based on data structures from the TCGplayer API (Catalog, Pricing, and Inventory endpoints).

## The Question
Can we predict the market price of a trading card based on its attributes? This tool helps collectors and sellers determine if a card is currently undervalued or overpriced.

## Dataset & Features
The data was modeled after TCGplayer's hierarchical structure:
- **Target Variable:** `market_price` (Continuous Numeric)
- **Features:** - `rarity_encoded` (Categorical converted to Ordinal)
    - `condition_score` (Mapping Mint to Damaged)
    - `popularity_score` (Proxy for market demand)
    - `set_name` (One-hot encoded)

## Model Selection
I chose a **Random Forest Regressor**. This model was selected because card pricing is often non-linear (e.g., the jump from "Rare" to "Ultra Rare" is not a simple addition) and Random Forests handle mixed feature types and outliers better than standard Linear Regression.

## Limitations & Bias
As discussed in the [Medium Post](YOUR_MEDIUM_LINK_HERE), the model struggles with "Limited Edition" cards or sudden "Hype Spikes" because the static dataset cannot account for real-time social media sentiment or tournament results.

## Setup
1. Install requirements: `pip install -r requirements.txt`
2. Run analysis: `python predict_prices.py`
