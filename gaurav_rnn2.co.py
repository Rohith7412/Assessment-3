import torch 
import torch.nn as nn
import torch.optim as optim

data = [
    ("i love this movie", 1),
    ("this film was great", 1),
    ("absolutely fantastic work", 1),
    ("i hated this movie", 0),
    ("this film was terrible", 0),
    ("absolutely worst experience", 0)
]

vocab = set()
for text, _ in data:
    for word in text.split():
        vocab.add(word)

word_to_idx = {word: idx + 1 for idx, word in enumerate(vocab)}
word_to_idx['<PAD>'] = 0
vocab_size = len(word_to_idx)

max_len = max(len(text.split()) for text, _ in data)

inputs = []
labels = []

for text, label in data:
    words = text.split()
    indexed = [word_to_idx[w] for w in words] + [0] * (max_len - len(words))
    inputs.append(indexed)
    labels.append(label)

X = torch.tensor(inputs, dtype=torch.long)
Y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

class SimpleTextRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SimpleTextRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, text):
        embedded = self.embedding(text)
        rnn_out, hidden = self.rnn(embedded)
        last_hidden = hidden.squeeze(0) 
        output = self.fc(last_hidden)
        return torch.sigmoid(output)

EMBEDDING_DIM = 8
HIDDEN_DIM = 16
MODEL = SimpleTextRNN(vocab_size, EMBEDDING_DIM, HIDDEN_DIM)

criterion = nn.BCELoss()
optimizer = optim.Adam(MODEL.parameters(), lr=0.01)

print("===== Launching new training optimizer ======")

for epoch in range(100):
    optimizer.zero_grad()
    predictions = MODEL(X)
    loss = criterion(predictions, Y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"epoch {epoch+1:02d}/100 | Training Loss: {loss.item():.4f}")


# ========================================================
# INTERACTIVE QUESTION & ANSWER LOOP (Stops code from terminating)
# ========================================================
print("\n ====== Running Evaluation Inference Check =======")
MODEL.eval()

with torch.no_grad():
    while True:
        # 1. Get input from the user
        test_sentence = input("\nEnter a sentence to analyze (or type 'quit' to exit): ").strip()
        
        # 2. Check for exit condition
        if test_sentence.lower() == 'quit':
            print("Exiting loop. Goodbye!")
            break
        
        if not test_sentence:
            continue

        # 3. Process the text to match training dimensions
        test_words = [word_to_idx.get(w, 0) for w in test_sentence.lower().split()]
        
        # Trim if user enters a sequence longer than max_len, or pad if shorter
        test_words = test_words[:max_len] + [0] * max(0, max_len - len(test_words))
        
        # 4. Predict
        test_tensor = torch.tensor([test_words], dtype=torch.long)
        prediction = MODEL(test_tensor).item()
        
        # 5. Output response
        verdict = 'Positive' if prediction > 0.5 else 'Negative'
        print(f"-> Predicted Sentiment Probability: {prediction:.4f}")
        print(f"-> Verdict: {verdict}")
