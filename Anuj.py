import torch
import torch.nn as nn
import torch.optim as optim


samples = [
    ("the movie was really good", 1),
    ("I enjoyed this film", 1),
    ("what a fantastic movie", 1),
    ("the movie was really bad", 0),
    ("I did not enjoy this film", 0),
    ("what a terrible experience", 0)
]


vocab = {"<PAD>": 0, "<UNK>": 1}

for sentence, _ in samples:
    for word in sentence.lower().split():
        if word not in vocab:
            vocab[word] = len(vocab)


print("Vocabulary size:", len(vocab))


def encode_sentence(sentence, size):
    words = sentence.lower().split()
    numbers = []

    for word in words:
        numbers.append(vocab.get(word, vocab["<UNK>"]))

    if len(numbers) < size:
        numbers.extend([vocab["<PAD>"]] * (size - len(numbers)))
    else:
        numbers = numbers[:size]

    return numbers


max_words = max(len(sentence.split()) for sentence, _ in samples)

X = torch.tensor(
    [encode_sentence(sentence, max_words) for sentence, _ in samples],
    dtype=torch.long
)

Y = torch.tensor(
    [[label] for _, label in samples],
    dtype=torch.float32
)


class SentimentRNN(nn.Module):

    def __init__(self, vocab_count, embed_size=10, memory_size=20):
        super().__init__()

        self.embedding = nn.Embedding(vocab_count, embed_size)

        self.rnn = nn.RNN(
            input_size=embed_size,
            hidden_size=memory_size,
            batch_first=True
        )

        self.output_layer = nn.Linear(memory_size, 1)

    def forward(self, x):

        x = self.embedding(x)

        _, hidden_state = self.rnn(x)

        hidden_state = hidden_state[-1]

        return self.output_layer(hidden_state)


model = SentimentRNN(len(vocab))

loss_function = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)


print("\nStarting training...")

epochs = 150

for step in range(epochs):

    model.train()

    optimizer.zero_grad()

    result = model(X)

    loss = loss_function(result, Y)

    loss.backward()

    optimizer.step()

    if (step + 1) % 25 == 0:
        print(
            f"Step {step + 1}/{epochs}  "
            f"Loss: {loss.item():.4f}"
        )


def check_sentiment(sentence):

    model.eval()

    encoded = encode_sentence(sentence, max_words)

    input_data = torch.tensor(
        [encoded],
        dtype=torch.long
    )

    with torch.no_grad():

        raw_output = model(input_data)

        probability = torch.sigmoid(raw_output).item()

    if probability >= 0.5:
        result = "Positive"
    else:
        result = "Negative"

    print("\nSentence:", sentence)
    print(f"Positive probability: {probability:.3f}")
    print("Result:", result)


print("\n--- Testing the model ---")

check_sentiment("I enjoyed this movie")
check_sentiment("the film was terrible")
check_sentiment("fantastic movie")