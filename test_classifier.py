
from finbert_classifier import classify_headline


print("🟢 Positive Headline:")
print(classify_headline("Apple stock surges after strong earnings report"))


print("🔴 Negative Headline:")
print(classify_headline("Tesla shares drop amid production concerns"))


print("⚪ Neutral Headline:")
print(classify_headline("The Federal Reserve holds interest rates steady"))

