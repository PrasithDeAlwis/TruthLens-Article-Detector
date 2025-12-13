import pandas as pd

# Load the dataset
df = pd.read_csv('data/balanced_2000_dataset.csv')

print("=" * 70)
print("DATASET INFORMATION FOR TRUTHLENS MODEL")
print("=" * 70)

print(f"\n📊 Dataset: balanced_2000_dataset.csv")
print(f"\n📈 Total Articles: {len(df):,}")
print(f"   ✓ Real News: {(df['label'] == 'real').sum():,}")
print(f"   ✓ Fake News: {(df['label'] == 'fake').sum():,}")
print(f"\n⚖️  Balance: {(df['label'] == 'real').sum() / len(df) * 100:.1f}% real, {(df['label'] == 'fake').sum() / len(df) * 100:.1f}% fake")

print(f"\n📝 Columns in Dataset:")
for col in df.columns:
    print(f"   - {col}")

print(f"\n📏 Text Length Statistics:")
print(f"   Real News (avg): {df[df['label']=='real']['text'].str.len().mean():.0f} characters")
print(f"   Fake News (avg): {df[df['label']=='fake']['text'].str.len().mean():.0f} characters")

print(f"\n📖 Word Count Statistics:")
print(f"   Real News (avg): {df[df['label']=='real']['text'].str.split().str.len().mean():.0f} words")
print(f"   Fake News (avg): {df[df['label']=='fake']['text'].str.split().str.len().mean():.0f} words")

print("\n" + "=" * 70)
print("SAMPLE ARTICLES")
print("=" * 70)

print("\n✅ REAL NEWS SAMPLE:")
real_sample = df[df['label'] == 'real'].iloc[10]['text']
print(f"   {real_sample[:200]}...")

print("\n❌ FAKE NEWS SAMPLE:")
fake_sample = df[df['label'] == 'fake'].iloc[10]['text']
print(f"   {fake_sample[:200]}...")

print("\n" + "=" * 70)
