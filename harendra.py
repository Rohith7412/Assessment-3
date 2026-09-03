import torch
import torch.nn as nn
import torch.optim as optim

data = [
    ("I love this movie", 1),
    ("this film was great", 1),
    ("absolutely fantastic work", 1),
    ("I hated this movie", 0),
    ("this film was terrible", 0),
    ("absolutely worst experience", 0)
]

vocab = set()
for text, _ in data:
    for word in text.split():
        vocab.add(word)

word_to_idx = {word: idx + 1 for idx, word in enumerate(vocab)}
word_to_idx["<PAD>"] = 0
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
Y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1) # Shape: (6, 1)


class SimpleTextRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SimpleTextRNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

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


print("--- Launching RNN Training Optimization ---")
for epoch in range(100):
    optimizer.zero_grad()           
    predictions = MODEL(X)            
    loss = criterion(predictions, Y)   
    loss.backward()                    
    optimizer.step()                   

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:03d}/100 | Training Loss Matrix Error: {loss.item():.4f}")


print("\n--- Running Evaluation Inference Check ---")
MODEL.eval()
with torch.no_grad():
    test_sentence = "I love fantastic film"

    test_words = test_sentence.split()

    test_indexed = [word_to_idx.get(w, 0) for w in test_words] + [0] * (max_len - len(test_words))

    test_tensor = torch.tensor([test_indexed], dtype=torch.long)
    prediction = MODEL(test_tensor).item()

    sentiment = "Positive" if prediction > 0.5 else "Negative"
    print(f"Input Text String: {test_sentence}")
    print(f"Raw Sigmoid Output Score: {prediction:.4f}")
    print(f"Determined Sentiment Classification: {sentiment}")