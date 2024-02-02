import torch
import sentencepiece
from transformers import AutoTokenizer, AutoModelWithLMHead
import pandas as pd

# GPU kiválasztása
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# A fordító inicializálása
model_name = 'Helsinki-NLP/opus-mt-en-hu'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelWithLMHead.from_pretrained(model_name).to(device)

# Függvény a fordításra
def translate(text):
    # Tokenizálás
    text = [text]
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)

    # Fordítás
    outputs = model.generate(**inputs)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return decoded

# A beszélgetések beolvasása CSV fájlból
input_file = 'twewy-name-line-full.csv'
df = pd.read_csv(input_file)

# A beszélgetések lefordítása
df["name"] = df["name"].apply(translate)
df["line"] = df["line"].apply(translate)

# A fordított adatok mentése CSV fájlba
output_file = 'output.csv'
df.to_csv(output_file, index=False)
