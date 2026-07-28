# SCT_ML_1
import numpy as np
import pandas as pd
import os

np.random.seed(42)

N=1000

sqft = np.random.normal(2000,700, N).clip(500, 6000)

bedrooms =np.clip((sqft/700) + np.random.normal(0,0.6,N),1,7).round().astype(float)

bathrooms=np.clip((bedrooms * 0.6)+(sqft/2500)+np.random.normal(0,0.5,N),1,5).round(1)


base_price = 50000
price_per_sqft = 120
bedroom_premium = 8000
bathroom_premium = 12000
noise = np.random.normal(0, 25000, N)

price = (
    base_price + (sqft * price_per_sqft) + (bedrooms * bedroom_premium) + (bathrooms * bathroom_premium) + noise).clip(50000, None)

df = pd.DataFrame({
    "sqft": sqft.round(0).astype(int),
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "price": price.round(0).astype(int)
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/house_prices.csv", index=False)

print(F"Generated data/house_prices.csv with {len(df)}rows")
print(df.head())
