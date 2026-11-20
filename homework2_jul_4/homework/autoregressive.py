import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 64, n_tokens: int = 2**10):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_latent = d_latent
        self.max_positions = 2048

        self.token_embedding = torch.nn.Embedding(n_tokens, d_latent)
        self.pos_embedding = torch.nn.Embedding(self.max_positions, d_latent)
        self.sos = torch.nn.Parameter(torch.zeros(1, 1, d_latent))
        self.gru = torch.nn.GRU(
            input_size=d_latent,
            hidden_size=d_latent,
            num_layers=1,
            batch_first=True,
        )
        self.lm_head = torch.nn.Linear(d_latent, n_tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        bsz, h, w = x.shape
        seq = x.view(bsz, h * w)
        seq_len = seq.shape[1]

        if seq_len > self.max_positions:
            raise ValueError(f"Sequence length {seq_len} exceeds max_positions={self.max_positions}")

        tok = self.token_embedding(seq)
        shifted = torch.cat([self.sos.expand(bsz, -1, -1), tok[:, :-1, :]], dim=1)

        pos = torch.arange(seq_len, device=x.device)
        shifted = shifted + self.pos_embedding(pos)[None, :, :]

        hidden, _ = self.gru(shifted)
        logits = self.lm_head(hidden).view(bsz, h, w, self.n_tokens)
        return logits, {}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        if device is None:
            device = next(self.parameters()).device

        seq_len = h * w
        if seq_len > self.max_positions:
            raise ValueError(f"Requested sequence length {seq_len} exceeds max_positions={self.max_positions}")

        tokens = torch.zeros(B, seq_len, device=device, dtype=torch.long)
        with torch.no_grad():
            hidden = None
            step_input = self.sos.expand(B, -1, -1) + self.pos_embedding.weight[0][None, None, :]
            for t in range(seq_len):
                step_output, hidden = self.gru(step_input, hidden)
                next_logits = self.lm_head(step_output[:, 0, :])
                next_probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(next_probs, num_samples=1).squeeze(-1)
                tokens[:, t] = next_token

                if t + 1 < seq_len:
                    step_input = self.token_embedding(next_token).unsqueeze(1)
                    step_input = step_input + self.pos_embedding.weight[t + 1][None, None, :]

        return tokens.view(B, h, w)
