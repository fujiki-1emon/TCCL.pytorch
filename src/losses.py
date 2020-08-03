import torch
from torch.nn import functional as F


def calculate_similarity(embeddings1, embeddings2, temperature):
    nc = embeddings1.size(1)
    # L2 distance
    emb1_norm = embeddings1.square().sum(dim=1)
    emb2_norm = embeddings2.square().sum(dim=1)
    dist = torch.max(
        emb1_norm + emb2_norm - 2.0 * torch.matmul(embeddings1, embeddings2.t()),
        torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
    )
    # similarity: (N, M)
    similarity = -1.0 * dist
    similarity /= embeddings1.size(1)
    similarity /= temperature
    return similarity


def embeddings_similarity(embeddings1, embeddings2, temperature):
    '''
    embeddings1: (N, D). in paper, U. u_i, i=1 to N
    embeddings2: (M, D). in paper, V. v_j, j=1 to M
    '''
    max_num_frames = embeddings1.size(0)

    # similarity
    nc = embeddings1.size(1)
    # L2 distance
    emb1_norm = embeddings1.square().sum(dim=1)
    emb2_norm = embeddings2.square().sum(dim=1)
    dist = torch.max(
        emb1_norm + emb2_norm - 2.0 * torch.matmul(embeddings1, embeddings2.t()),
        torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
    )
    # similarity: (N, M)
    similarity = -1.0 * dist
    similarity /= embeddings1.size(1)
    similarity /= temperature
    similarity = F.softmax(similarity, dim=1)
    # v_tilda
    soft_nearest_neighbor = torch.matmul(similarity, embeddings2)

    # logits for Beta_k
    logits = calculate_similarity(soft_nearest_neighbor, embeddings1, temperature)
    labels = F.one_hot(torch.tensor(range(max_num_frames)), num_classes=max_num_frames)

    return logits, labels


def cycleback_regression_loss(
    logits, labels, num_frames, steps, seq_lens, normalize_indices, variance_lambda, args,
):
    labels = labels.to(args.device)
    labels = labels.detach()  # (bs, ts)
    steps = steps.detach()  # (bs, ts)
    steps = steps.float()
    seq_lens = seq_lens.float()

    seq_lens = seq_lens.unsqueeze(1).repeat(1, num_frames)
    steps = steps / seq_lens

    beta = F.softmax(logits, dim=1)
    true_timesteps = (labels * steps).sum(dim=1)  # (bs, )
    pred_timesteps = (beta * steps).sum(dim=1)  # (bs, )  # sum_{k=1 to N}(beta_k * k)
    pred_timesteps_repeated = pred_timesteps.unsqueeze(1).repeat(1, num_frames)
    pred_timesteps_var = ((steps - pred_timesteps_repeated).square() * beta).sum(dim=1)
    pred_timesteps_log_var = pred_timesteps_var.log()
    squared_error = (true_timesteps - pred_timesteps).square()
    loss = torch.mean((-pred_timesteps_log_var).exp() * squared_error + variance_lambda * pred_timesteps_log_var)
    return loss


def temporal_cycle_consistency_loss(
    embeddings, steps, seq_lens, num_frames, batch_size, temperature, variance_lambda, normalize_indices,
    args,
):
    logits_list = []
    labels_list = []
    steps_list = []
    seq_lens_list = []
    for i in range(batch_size):
        for j in range(batch_size):
            if i != j:
                logits, labels = embeddings_similarity(embeddings[i], embeddings[j], temperature)
                logits_list.append(logits)
                labels_list.append(labels)
                steps_list.append(steps[i:i+1].repeat(num_frames, 1))  # (N, T) -> (1, T) -> (T, T)  # TODO?
                seq_lens_list.append(seq_lens[i:i+1].repeat(num_frames))  # (N,) -> (1,) -> (T,)
    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    steps = torch.cat(steps_list, dim=0)
    seq_lens = torch.cat(seq_lens_list, dim=0)

    loss = cycleback_regression_loss(
        logits, labels, num_frames, steps, seq_lens, normalize_indices, variance_lambda,
        args,
    )

    return loss
