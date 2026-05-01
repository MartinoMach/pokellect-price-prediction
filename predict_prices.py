import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- 1. DATA COLLECTION (TCGplayer API Logic) ---
def get_tcgplayer_data():
    """
    Simulates fetching data from Catalog and Pricing endpoints.
    Structure: productId -> rarityName, groupName (Set), condition, marketPrice.
    """
    # Generating a larger synthetic dataset to represent the 'collected' data
    np.random.seed(42)
    data_size = 200
    
    rarities = ['Common', 'Uncommon', 'Rare', 'Ultra Rare', 'Secret Rare']
    sets = ['Base Set', 'Jungle', 'Fossil', 'Team Rocket', 'Neo Genesis']
    conditions = ['Mint', 'Near Mint', 'Lightly Played', 'Moderately Played', 'Damaged']
    
    base_data = {
        'card_name': [f"Card_{i}" for i in range(data_size)],
        'rarity': np.random.choice(rarities, data_size),
        'set_name': np.random.choice(sets, data_size),
        'condition': np.random.choice(conditions, data_size),
        'popularity_score': np.random.randint(1, 100, data_size),
        'release_year': np.random.choice([1999, 2000, 2001, 2002], data_size)
    }
    
    df = pd.DataFrame(base_data)
    
    # Logic-based Pricing: Price = (Rarity * 10) + (Popularity * 0.5) - (Age * 2) + Noise
    # This ensures the model HAS something to learn.
    rarity_map = {'Common': 1, 'Uncommon': 2, 'Rare': 5, 'Ultra Rare': 20, 'Secret Rare': 50}
    df['market_price'] = df['rarity'].map(rarity_map) * 5 + (df['popularity_score'] * 0.3)
    df['market_price'] += np.random.normal(0, 5, data_size) # Add variance
    df['market_price'] = df['market_price'].clip(lower=2.0) # No negative prices

    # --- INJECTING YOUR MEDIUM POST SAMPLES ---
    # We manually add the 5 samples you discussed so the code matches your story.
    outliers = pd.DataFrame([
        {'card_name': 'Card A', 'rarity': 'Ultra Rare', 'set_name': 'Base Set', 'condition': 'Near Mint', 'popularity_score': 99, 'release_year': 1999, 'market_price': 25.0},
        {'card_name': 'Card B', 'rarity': 'Rare', 'set_name': 'Jungle', 'condition': 'Mint', 'popularity_score': 10, 'release_year': 1998, 'market_price': 5.0},
        {'card_name': 'Card C', 'rarity': 'Secret Rare', 'set_name': 'Neo Genesis', 'condition': 'Near Mint', 'popularity_score': 85, 'release_year': 2000, 'market_price': 50.0},
        {'card_name': 'Card D', 'rarity': 'Common', 'set_name': 'Fossil', 'condition': 'Damaged', 'popularity_score': 5, 'release_year': 1999, 'market_price': 2.0},
        {'card_name': 'Card E', 'rarity': 'Ultra Rare', 'set_name': 'Promo', 'condition': 'Mint', 'popularity_score': 95, 'release_year': 2001, 'market_price': 100.0}
    ])
    
    return pd.concat([df, outliers], ignore_index=True)

# --- 2. DATA CLEANING & PREPROCESSING ---
def preprocess(df):
    # Mapping Rarity and Condition to numeric scores for the Regressor
    rarity_map = {'Common': 1, 'Uncommon': 2, 'Rare': 3, 'Ultra Rare': 4, 'Secret Rare': 5}
    condition_map = {'Mint': 5, 'Near Mint': 4, 'Lightly Played': 3, 'Moderately Played': 2, 'Damaged': 1}
    
    df['rarity_encoded'] = df['rarity'].map(rarity_map)
    df['condition_encoded'] = df['condition'].map(condition_map)
    
    # One-hot encoding for Set names
    df_final = pd.get_dummies(df, columns=['set_name'])
    
    # Cleaning nulls (if any)
    df_final = df_final.dropna()
    
    return df_final

# --- 3. MODEL TRAINING & EVALUATION ---
def train_and_evaluate(df):
    # Features vs Target
    # We drop non-numeric/identifier columns
    X = df.drop(columns=['card_name', 'rarity', 'condition', 'market_price'])
    y = df['market_price']
    
    # Split
    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        X, y, df['card_name'], test_size=0.2, random_state=42
    )
    
    # Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    preds = model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    print("--- Model Performance ---")
    print(f"Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
    
    # Error Analysis
    results = pd.DataFrame({
        'Card': names_test,
        'Actual': y_test,
        'Predicted': preds,
        'Error': abs(y_test - preds)
    })
    
    # Displaying the 5 most "wrong" cards
    print("\n--- Top 5 Misclassified Samples (Error Analysis) ---")
    print(results.sort_values(by='Error', ascending=False).head(5))

if __name__ == "__main__":
    raw_data = get_tcgplayer_data()
    processed_data = preprocess(raw_data)
    train_and_evaluate(processed_data)
