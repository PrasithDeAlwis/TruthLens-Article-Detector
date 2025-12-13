import pandas as pd

df = pd.read_csv('data/balanced_2000_dataset.csv')

print("=" * 70)
print("ANALYZING DATASET QUALITY")
print("=" * 70)

# Show some real examples
print("\n📰 REAL NEWS EXAMPLES:")
print("-" * 70)
real_samples = df[df['label'] == 'real'].head(20)
for i, row in real_samples.iterrows():
    print(f"{i+1}. {row['text']}")

print("\n" + "=" * 70)
print("\n🚫 FAKE NEWS EXAMPLES:")
print("-" * 70)
fake_samples = df[df['label'] == 'fake'].head(10)
for i, row in fake_samples.iterrows():
    print(f"{i+1}. {row['text'][:200]}...")
    print()

print("\n" + "=" * 70)
print("LENGTH DISTRIBUTION:")
print("-" * 70)
print(f"Real news - Min: {df[df['label']=='real']['text'].str.len().min()}")
print(f"Real news - Max: {df[df['label']=='real']['text'].str.len().max()}")
print(f"Real news - Median: {df[df['label']=='real']['text'].str.len().median()}")
print(f"\nFake news - Min: {df[df['label']=='fake']['text'].str.len().min()}")
print(f"Fake news - Max: {df[df['label']=='fake']['text'].str.len().max()}")
print(f"Fake news - Median: {df[df['label']=='fake']['text'].str.len().median()}")
