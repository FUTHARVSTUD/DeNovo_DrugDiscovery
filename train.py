import os
import torch.distributed
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import selfies as sf
from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors
from rdkit.Chem import Lipinski
import argparse
from torch.cuda.amp import GradScaler, autocast

def parse_args():
    parser = argparse.ArgumentParser(description="Train conditional WGAN-GP on SELFIES")
    parser.add_argument("--data_path", type=str, default="~/project/DeNovo_DrugDiscovery/zinc_selfies_with_props.csv")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--noise_dim", type=int, default=128)
    parser.add_argument("--prop_dim", type=int, default=6)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--n_critic", type=int, default=5)
    parser.add_argument("--lambda_gp", type=float, default=15.0)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="~/projects/DeNovo_DrugDiscovery/outputs/exp1")
    parser.add_argument("--checkpoint_interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau_start", type=float, default=2.0)
    parser.add_argument("--tau_end",   type=float, default=0.5)
    parser.add_argument("--tau_anneal_epochs", type=int, default=150)
    return parser.parse_args()

def main(args):
    # Reproducibility: set random seeds
    seed = args.seed if hasattr(args, "seed") else 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Force cuDNN to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Distributed setup
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", index=args.local_rank)

    data_path = args.data_path
    df = pd.read_csv(data_path)
    if args.local_rank == 0:
        print("Loaded", len(df), "molecules.")

    # Tokenize SELFIES and build vocabulary
    def tokenize_selfies(selfies_str):
        return list(sf.split_selfies(selfies_str))

    all_tokens = []
    for s in df['SELFIES']:
        tokens = tokenize_selfies(s)
        all_tokens.extend(tokens)

    # Build vocabulary (add special tokens)
    special_tokens = ["<PAD>", "<SOS>", "<EOS>"]
    vocab = special_tokens + sorted(set(all_tokens))
    token2idx = {t: i for i, t in enumerate(vocab)}
    idx2token = {i: t for t, i in token2idx.items()}
    vocab_size = len(vocab)
    if args.local_rank == 0:
        print("Vocab size:", vocab_size)

    #%% [code]
    # Create a PyTorch Dataset for conditional generation.
    # Each sample: (SELFIES sequence as indices, property vector)
    class SelfiesConditionalDataset(Dataset):
        def __init__(self, df, token2idx, max_len=64):
            self.df = df
            self.token2idx = token2idx
            self.max_len = max_len

            # Define the property columns you want to condition on
            self.prop_cols = ['QED', 'LogP', 'Molecular_Weight', 'Lipinski_H_Bond_Donors', 'Lipinski_H_Bond_Acceptors', 'Lipinski_Rule_Violation']

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            selfies_str = row['SELFIES']
            tokens = ["<SOS>"] + tokenize_selfies(selfies_str) + ["<EOS>"]
            # Pad token sequence
            if len(tokens) < self.max_len:
                tokens += ["<PAD>"] * (self.max_len - len(tokens))
            else:
                tokens = tokens[:self.max_len]
            token_ids = [self.token2idx[t] for t in tokens]
            token_ids = torch.tensor(token_ids, dtype=torch.long)

            # Get property vector
            prop_vec = torch.tensor(row[self.prop_cols].astype(float).values, dtype=torch.float32)
            return token_ids, prop_vec

    dataset = SelfiesConditionalDataset(df, token2idx, max_len=args.max_len)
    # Distributed sampler for DataLoader
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    if args.local_rank == 0:
        print("Dataset prepared, number of batches:", len(dataloader))

    #%% [code]
    # === 2. Model Definition ===
    # We'll define a conditional Generator and a conditional Discriminator.
    # The generator takes noise and a property vector as input, and outputs a sequence of logits over tokens.
    # We use the Gumbel-Softmax trick to approximate sampling from a discrete distribution.

    def gumbel_softmax_sample(logits, tau=1.0, eps=1e-10):
        U = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(U + eps) + eps)
        y = logits + gumbel
        return F.softmax(y / tau, dim=-1)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.local_rank == 0:
        print(f"Using device: {device}")

    class ConditionalGenerator(nn.Module):
        def __init__(self, noise_dim, prop_dim, hidden_dim, vocab_size, max_len, embed_dim=256):
            super(ConditionalGenerator, self).__init__()
            self.noise_dim = noise_dim
            self.prop_dim = prop_dim
            self.hidden_dim = hidden_dim
            self.max_len = max_len
            self.vocab_size = vocab_size

            # Map noise and property vector to initial hidden state
            self.fc = nn.Linear(noise_dim + prop_dim, hidden_dim)

            # Embedding and LSTM decoder
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
            self.fc_out = nn.Linear(hidden_dim, vocab_size)

        def forward(self, noise, prop_vec, temperature=1.0):
            # noise: [batch, noise_dim]
            # prop_vec: [batch, prop_dim]
            batch_size = noise.size(0)
            # Concatenate noise and property vector, and map to hidden state
            h0 = torch.tanh(self.fc(torch.cat([noise, prop_vec], dim=1)))  # [batch, hidden_dim]
            h0 = h0.unsqueeze(0)  # [1, batch, hidden_dim]
            c0 = torch.zeros_like(h0)

            # Start token (<SOS>) for every sample
            sos_token = token2idx["<SOS>"]
            inputs = torch.full((batch_size, 1), sos_token, dtype=torch.long, device=noise.device)

            outputs = []
            hidden, cell = h0, c0
            for t in range(self.max_len - 1):
                emb = self.embedding(inputs)  # [batch, 1, embed_dim]
                out, (hidden, cell) = self.lstm(emb, (hidden, cell))  # out: [batch, 1, hidden_dim]
                logits = self.fc_out(out.squeeze(1))  # [batch, vocab_size]
                # Gumbel-Softmax sampling (differentiable approximation)
                probs = gumbel_softmax_sample(logits, tau=temperature, eps=1e-10)
                outputs.append(probs.unsqueeze(1))  # store for later loss calculation
                # For next time step, take the argmax (non-differentiable but used only in inference)
                inputs = torch.argmax(probs, dim=-1, keepdim=True)
            # Concatenate outputs along the sequence length dimension
            outputs = torch.cat(outputs, dim=1)  # [batch, max_len-1, vocab_size]
            return outputs

    class ConditionalDiscriminator(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_dim, prop_dim, max_len):
            super(ConditionalDiscriminator, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim + prop_dim, 1)
            self.max_len = max_len

        def forward(self, token_ids, prop_vec):
            # token_ids: [batch, max_len] (discrete indices)
            emb = self.embedding(token_ids)  # [batch, max_len, embed_dim]
            _, (hidden, _) = self.lstm(emb)     # hidden: [1, batch, hidden_dim]
            hidden = hidden.squeeze(0)          # [batch, hidden_dim]
            x = torch.cat([hidden, prop_vec], dim=1)  # [batch, hidden_dim+prop_dim]
            out = self.fc(x)                    # [batch, 1]
            return out

        def forward_from_emb(self, emb, prop_vec):
            # emb: continuous embeddings, shape [batch, max_len, embed_dim]
            _, (hidden, _) = self.lstm(emb)     # hidden: [1, batch, hidden_dim]
            hidden = hidden.squeeze(0)          # [batch, hidden_dim]
            x = torch.cat([hidden, prop_vec], dim=1)  # [batch, hidden_dim+prop_dim]
            out = self.fc(x)                    # [batch, 1]
            return out

    #%% [code]
    # Hyperparameters
    noise_dim = args.noise_dim
    prop_dim = args.prop_dim   # number of conditioning properties
    hidden_dim = args.hidden_dim
    embed_dim = args.embed_dim
    max_len = args.max_len

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = ConditionalGenerator(noise_dim, prop_dim, hidden_dim, vocab_size, max_len, embed_dim).to(device)
    D = ConditionalDiscriminator(vocab_size, embed_dim, hidden_dim, prop_dim, max_len).to(device)
    # Wrap models with DDP
    G = nn.parallel.DistributedDataParallel(G, device_ids=[args.local_rank], output_device=args.local_rank)
    D = nn.parallel.DistributedDataParallel(D, device_ids=[args.local_rank], output_device=args.local_rank)

    #%% [code]
    # === 3. Losses and Optimizers ===
    lr = args.lr
    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.9))

    # lr decay for scheduler
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=50, gamma=0.5)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=50, gamma=0.5)

    # WGAN-GP hyperparameters
    lambda_gp = args.lambda_gp

    def gradient_penalty(D, real_samples, fake_samples, prop_vec):
        # Unwrap DistributedDataParallel to access module attributes
        if isinstance(D, torch.nn.parallel.DistributedDataParallel):
            disc = D.module
        else:
            disc = D
        batch_size = real_samples.size(0)
        # Obtain embeddings from the discriminator's embedding layer
        real_emb = disc.embedding(real_samples)   # shape: [batch, max_len, embed_dim]
        fake_emb = disc.embedding(fake_samples)     # shape: [batch, max_len, embed_dim]

        # Create alpha with shape [batch, 1, 1] and expand to match real_emb
        alpha = torch.rand(batch_size, 1, 1, device=device)
        alpha = alpha.expand_as(real_emb)  # shape: [batch, max_len, embed_dim]

        # Interpolate in the embedding space
        interpolates = alpha * real_emb + (1 - alpha) * fake_emb
        interpolates.requires_grad_(True)

        # Disable cuDNN for the RNN forward pass in the gradient penalty calculation
        with torch.backends.cudnn.flags(enabled=False):
            d_interpolates = disc.forward_from_emb(interpolates, prop_vec)

        fake = torch.ones(d_interpolates.size(), device=device)

        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.reshape(batch_size, -1)
        grad_norm = gradients.norm(2, dim=1)
        gp = ((grad_norm - 1) ** 2).mean()
        return gp

    #%% [code]
    # === 4. Reinforcement Learning Reward Component ===
    # Here we define a property reward function. For example, we can use QED.
    # In practice you might combine several metrics.
    def compute_property_reward(selfies_probs):
        """
        selfies_probs: [batch, max_len-1, vocab_size] output from G (Gumbel-softmax probabilities)
        To compute the reward, we need to decode the probabilities to SELFIES.
        Here, for simplicity, we take argmax at each time step.
        """
        batch_size = selfies_probs.size(0)
        # Get token indices from probabilities
        token_ids = torch.argmax(selfies_probs, dim=-1).cpu().numpy()  # shape [batch, max_len-1]
        rewards = []
        for seq in token_ids:
            # Convert indices to tokens (skip <SOS>)
            tokens = [idx2token[idx] for idx in seq if idx != token2idx["<PAD>"]]
            # Stop at <EOS>
            if "<EOS>" in tokens:
                tokens = tokens[:tokens.index("<EOS>")]
            selfies_str = "".join(tokens)
            try:
                smi = sf.decoder(selfies_str)
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    # For example, reward can be the QED score
                    reward = QED.qed(mol)
                else:
                    reward = 0.0
            except Exception as e:
                reward = 0.0
            rewards.append(reward)
        return torch.tensor(rewards, dtype=torch.float32, device=device)

    #%% [code]
    # === 5. Training Loop ===
    num_epochs = args.num_epochs  # Adjust based on your dataset and training needs
    n_critic = args.n_critic      # Number of D updates per G update
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'training_logs')) if args.local_rank == 0 else None

    scaler_D = GradScaler()
    scaler_G = GradScaler()

    for epoch in range(1, num_epochs+1):
        # Shuffle sampler for each epoch
        sampler.set_epoch(epoch)
        tau = max(
        args.tau_end,
        args.tau_start
        - (args.tau_start - args.tau_end) * epoch / args.tau_anneal_epochs
        )

        for i, (real_seq, prop_vec) in enumerate(dataloader):
            batch_size = real_seq.size(0)
            real_seq = real_seq.to(device)
            prop_vec = prop_vec.to(device)

            # Train Discriminator
            optimizer_D.zero_grad()

            with autocast():
                # Sample noise and generate fake sequences from G
                noise = torch.randn(batch_size, noise_dim, device=device)
                fake_probs = G(noise, prop_vec, temperature=tau)  # shape: [batch, max_len-1, vocab_size]

                # For D, we need discrete tokens.
                # Here we use argmax for real/fake samples (note: not differentiable, but only used for D)
                fake_seq = torch.argmax(fake_probs, dim=-1)  # [batch, max_len-1]
                # Pad fake_seq to max_len by adding a PAD token at the end if necessary.
                fake_seq = F.pad(fake_seq, (0, 1), value=token2idx["<PAD>"])

                # Discriminator outputs
                d_real = D(real_seq, prop_vec)
                d_fake = D(fake_seq.detach(), prop_vec)

                # Wasserstein loss and gradient penalty
                d_loss = -torch.mean(d_real) + torch.mean(d_fake)
                gp = gradient_penalty(D, real_seq, fake_seq.detach(), prop_vec)
                d_loss_total = d_loss + lambda_gp * gp

            scaler_D.scale(d_loss_total).backward()
            scaler_D.step(optimizer_D)
            scaler_D.update()

            # Train Generator every n_critic iterations
            if i % n_critic == 0:
                optimizer_G.zero_grad()
                with autocast():
                    noise = torch.randn(batch_size, noise_dim, device=device)
                    fake_probs = G(noise, prop_vec, temperature=tau)
                    fake_seq = torch.argmax(fake_probs, dim=-1)
                    fake_seq = F.pad(fake_seq, (0, 1), value=token2idx["<PAD>"])

                    # Standard generator loss (Wasserstein)
                    g_loss_adv = -torch.mean(D(fake_seq, prop_vec))

                    # Compute property reward from generated SELFIES
                    prop_reward = compute_property_reward(fake_probs)
                    prop_reward = (prop_reward - 0.5) * 2
                    valid_count = (prop_reward > -0.9).sum().item()  # near 0 = invalid
                    valid_percent = valid_count / batch_size * 100
                    # We want to reward high QED, so we subtract the reward to reduce loss.
                    reward_scale = 0.5  # can tune this over time
                    g_loss = g_loss_adv - reward_scale * torch.mean(prop_reward)

                scaler_G.scale(g_loss).backward()
                scaler_G.unscale_(optimizer_G)
                torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
                scaler_G.step(optimizer_G)
                scaler_G.update()

        # Logging
        if args.local_rank == 0:
            writer.add_scalar("Loss/Discriminator", d_loss_total.item(), epoch)
            writer.add_scalar("Loss/Generator", g_loss.item(), epoch)
            writer.add_scalar("PropertyReward", torch.mean(prop_reward).item(), epoch)
            writer.add_scalar("Reward/ValidPercent", valid_percent, epoch)
            print(f"Valid molecules: {valid_count}/{batch_size} ({valid_percent:.2f}%)")
            print(f"Epoch [{epoch}/{num_epochs}] d_loss: {d_loss_total.item():.4f} | g_loss: {g_loss.item():.4f} | Reward: {torch.mean(prop_reward).item():.4f}")

        # Optionally, save checkpoints every few epochs
        torch.distributed.barrier()
        if epoch % args.checkpoint_interval == 0:
            if args.local_rank == 0:
                checkpoint_path = os.path.join(args.output_dir, "Checkpoints", f"checkpoint_epoch_{epoch}.pth")
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'G_state_dict': G.state_dict(),
                    'D_state_dict': D.state_dict(),
                    'optimizer_G_state_dict': optimizer_G.state_dict(),
                    'optimizer_D_state_dict': optimizer_D.state_dict()
                }, checkpoint_path)
                print(f"Checkpoint saved at epoch {epoch}")
                scheduler_G.step()
                scheduler_D.step()

    if args.local_rank == 0 and writer is not None:
        writer.close()

if __name__ == "__main__":
    args = parse_args()
    main(args)