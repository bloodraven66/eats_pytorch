import torch
import torch.nn as nn

'''
NOTES
~~~~~
- No class conditioned BN (speaker embed + noise)

'''

'''
Author: Sathvik Udupa
Reference: https://arxiv.org/pdf/2006.03575.pdf
'''

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        return



class Aligner(nn.Module):
    def __init__(self,
                num_embed,
                embed_dim,
                kernel_size,
                padding,
                dilations,
                num_blocks
                ):
        super().__init__()
        self.token_embed = nn.Embedding(num_embed, embed_dim)
        blocks = []
        for _ in range(num_blocks):
            layers = []
            for a, b in dilations:
                layers.append(nn.ReLU())
                layers.append(nn.Conv1d(embed_dim, embed_dim, kernel_size, dilation=a, padding=a))
                layers.append(nn.ReLU())
                layers.append(nn.Conv1d(embed_dim, embed_dim, kernel_size, dilation=b, padding=b))
            blocks.append(nn.Sequential(*layers))
        self.blocks = nn.Sequential(*blocks)
        self.relu = nn.ReLU()
        self.conv1x1_1 = nn.Conv1d(embed_dim, embed_dim, 1)
        self.conv1x1_2 = nn.Conv1d(embed_dim, 1, 1)

    def forward(self, x, lengths, out_offset, out_sequence_length=400):
        x = self.token_embed(x).permute(0, 2, 1)
        x = self.blocks(x)
        x = self.relu(x)
        x = self.conv1x1_1(x)
        x = self.relu(x)
        x = self.conv1x1_2(x)
        x = torch.randn((4, 1, 600))+1
        token_lengths = self.relu(x.squeeze())
        token_ends = torch.cumsum(token_lengths, 1)
        token_centres = token_ends - (token_lengths / 2.)

        aligned_lengths = [end[length-1] for end, length in zip(token_ends, lengths)]
        print(aligned_lengths, lengths)
        return x

    def sample(self, x, lengths, out_sequence_length=6000):
        return

dim = 256
vocab = 40
kernel = 3
padding = 1
dilations = [(1, 2), (4, 8), (16, 32)]
num_blocks = 2


aligner = Aligner(vocab, dim, kernel, padding, dilations, num_blocks)

data = torch.randint(low=0, high=39, size=(4, 400))
lengths = torch.randint(low=200, high=400, size=(4, 1))
out_offset = torch.randint(0, torch.min(lengths)-200)
# print(aligner)

print('input shape:', data.shape)

out = aligner(data, lengths)
print('output shape:', out.shape)
