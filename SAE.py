import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        seq_len = 264 #200
        self.fc = nn.Linear(seq_len*hidden_dim, z_dim)

    def forward(self, x):
        #print("in", x.size())
        out = x.permute(1, 0, 2)
        #print("permute", out.size())
        out = self.transformer(x) 
        #print("transformer", out.size())
        #out = out.permute(1, 0, 2)
        #print("permuted back", out.size())
        out = out.reshape(x.size(0), x.size(1) * x.size(2))
        out = self.fc(out)
        #out = torch.rand(out.size(0), out.size(1)).cuda()
        #out = torch.mean(out, 1)
        #print("out", out.size())
        return out

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim):
        super().__init__()
        seq_len = 264 #200
        self.fc = nn.Linear(z_dim, seq_len*hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(hidden_dim, nhead=1)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.tgt_mask = None
        self.mem_mask = None

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _generate_subsequent_mask(self, tgt_sz, src_sz):
            mask = (torch.triu(torch.ones(src_sz, tgt_sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            return mask

    def forward(self, z, tgt):
        # retrieve the decoder output from the compressed vector
        z = self.fc(z).reshape(tgt.size(0), tgt.size(1)-1, tgt.size(2))
        
        # shape (seq_len, batch_size, hidden_dim) for the transformer
        tgt = tgt.permute(1,0,2)
        z = z.permute(1,0,2)
        
        # generate the transformer masks
        tgt_len = tgt.size(0)
        mem_len = z.size(0)
        if self.tgt_mask is None or self.tgt_mask.size(0) != tgt_len:
            device = tgt.device
            mask = self._generate_square_subsequent_mask(tgt_len).to(device)
            self.tgt_mask = mask
        if self.mem_mask is None or self.mem_mask.size(0) != mem_len:
            device = z.device
            mask = self._generate_subsequent_mask(tgt_len, mem_len).to(device)
            self.mem_mask = mask

        # run the transformer
        out = self.transformer(tgt=tgt, memory=z, tgt_mask=self.tgt_mask, memory_mask=self.mem_mask)
        out = out.permute(1,0,2)
        out = self.out(out)
        return out


class SAE(nn.Module):
    """Defines the sequential autoencoder for DNA sequences"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, z_dim, n_layers, batch_first=True, dropout=0):
        super(SAE, self).__init__()
        # Set params
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        # Set modules
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = Encoder(embed_dim, hidden_dim, z_dim)
        self.decoder = Decoder(z_dim, hidden_dim, vocab_size)

    def forward(self, x, tgt):
        emb = self.embedding(x) # get embedding
        tgt_emb = self.embedding(tgt)
        z = self.encoder(emb)
        #z = torch.rand(x.size(0), x.size(1), 6).cuda()
        out = self.decoder(z, tgt_emb)
        return out

    def forward_encoder(self, x):
        """Only passes the input through the encoder and returns the result"""
        emb = self.embedding(x) # get embedding
        z = self.encoder(emb)
        return z

class OurDataset(Dataset):
    """Dataset of DNA sequences without labels for unsupervised learning"""
    def __init__(self, fname):
        """ 
        Args:
            fname (string): Path to the csv file with the unlabelled data, one sequence per row
        """
        print("Loading dataset...", end=" ")
        self.data = pd.read_csv(fname, sep="\t", encoding="utf-8")
        print("Done")
        print("Building vocab...", end=" ")
        self.vocab = {"<SOS>": 0, "<EOS>": 1}
        n_vocab = 2
        for index, sequence in self.data["Sequence"].iteritems():
            for word in sequence.split():
                if word in self.vocab:
                    continue
                else:
                    self.vocab[word] = n_vocab
                    n_vocab = n_vocab + 1
        self.vocab_length = len(self.vocab) + 1
        print("Done, vocab size:", self.vocab_length)

    def __word_encoding(self, word):
        """Returns encoding of a single word using self.vocab
        
        Args:
            word (string): word to retrieve the encoding for
        """
        return self.vocab.get(word, len(self.vocab)) # Returns the highest encoding in the dictionary + 1 for unknown characters

    def __get_encoding(self, sequence):
        """Takes a string of characters and returns the index encoding 
        
        Args:
            sequence (string): sequence of characters to retrieve index encoding for
        """
        encoding = []
        for idx, word in enumerate(sequence.split()):
            enc = self.__word_encoding(word)
            encoding.append(enc)
        encoding = torch.tensor(encoding).to(device)
        return encoding

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """Returns a tuple of an encoded item from the dataset located at given index along with its name like so: (encoding, name)
        Args:
            idx (integer): index of the item to be retrieved
        """
        name, sequence = self.data.iloc[idx]
        sequence = sequence.upper() # only use uppercase
        embedding = self.__get_encoding(sequence)
        return embedding, name

    def pad_collate(self, batch):
        """A custom variant of callate_fn that pads according to the longest sequence in a batch of sequences. This function is called by the dataloader. Returns the padded sequences, their labels and the original sequence lengths before padding.
        Args:
            batch (list): list of sequences to pad where each item is a tuple (encoding, name)
        """
        batch, names = map(list,zip(*batch))
        batch.sort(key=lambda x:len(x), reverse=True) # Sort by length!
        lengths = torch.tensor([t.shape[0] for t in batch]).to(device)
        batch = pad_sequence(batch, batch_first=True)
        return [batch, lengths, names]

class Model:
    def __init__(self):
        print("Initializing model...", end=" ")
        # Parameters
        vocab_size = 51327
        embed_dim = 200
        hidden_dim = 200
        z_dim = 30
        # Model
        self.net = SAE(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, z_dim=z_dim, n_layers=1, dropout=0) # set up the NN
        if torch.cuda.is_available():
            self.net.cuda() # Enable cuda
        self.opt = optim.Adam(self.net.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4) # set up the optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = 32
        self.dataset = None # training set
        self.train_loader = None # data loader for training set
        self.inference_loader = None # data loader for validation set
        print("Done")

    def load(self, fname):
        self.dataset = OurDataset(fname) # create our dataset from file
        self.train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.dataset.pad_collate, num_workers=0)
    def train_epoch(self):
        """Trains the model for one epoch"""
        losses = []

        self.net.train()

        # Training
        for i, data in enumerate(self.train_loader, 0):
            batch, batch_lengths, names = data
            batch_size = batch.size(0)
            tgt = torch.cat((torch.zeros(batch_size, 1, dtype=torch.long).cuda(), batch), dim=1)
            # (0) Zero gradients
            self.opt.zero_grad()
            # (1) Forward pass
            reconstruction = self.net(batch, tgt).permute(0,2,1)
            # (2) Compute diff
            y = torch.cat((batch, torch.ones(batch_size, 1, dtype=torch.long).cuda()), dim=1)
            loss = self.criterion(reconstruction, y)
            losses.append(loss.data.cpu().numpy())
            #print("Current loss:", loss.data.cpu().numpy())
            # (3) Compute gradients
            loss.backward()
            # (4) Update weights
            self.opt.step()
            
        self.net.eval()

        mean_loss = np.mean(losses)
        return mean_loss

    def train(self, num_epochs):
        """Trains the model.
        Args:
            num_epochs (integer): number of epochs to train the model for
        """
        print("Beginning training...")
        for e in range(num_epochs):
            loss = self.train_epoch()
            #print("Epoch", e+1, "of", num_epochs, "Training loss:", loss, "Validation loss:", val_loss)
            print("Epoch", e+1, "of", num_epochs, "Loss:", loss) 

    def inference(self):
        self.net.eval()
        labels = []
        latents = []
        for i, data in enumerate(self.train_loader, 0):
            batch, batch_lengths, names = data
            # Forward pass through the encoder only to obtain latent vector
            z = self.net.forward_encoder(batch).detach().cpu().numpy()
            for j, name in enumerate(names):
                labels.append(name)
                latents.append(z[j])
        df = pd.DataFrame(list(zip(labels, latents)), columns=['Name', 'z'])
        df.to_pickle("latents.pickle")
        print(df)

# Use GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
torch.backends.cudnn.benchmark=True

# Create Model
model = Model()
model.load("covid_unaligned_full.csv")
model.train(num_epochs=50)
model.inference()
