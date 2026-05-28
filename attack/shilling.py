import numpy as np
from scipy.sparse import vstack, csr_matrix


def inject_shilling_attack(trn_mat, item_popularity, n_genuine_users, attack_cfg):
    """
    Append synthetic shilling-attack users to the training matrix.

    Returns
    -------
    poisoned_coo : scipy.sparse.coo_matrix  (n_genuine + n_fake) × n_items
    target_items : list[int]                 fixed targets (same for all seeds/variants)
    n_fake       : int
    """
    attack_size  = attack_cfg['attack_size']
    num_targets  = attack_cfg['num_targets']
    strategy     = attack_cfg.get('strategy', 'bandwagon')
    target_seed  = attack_cfg.get('target_seed', 42)
    filler_size  = attack_cfg.get('filler_size', None)

    n_items  = trn_mat.shape[1]
    n_fake   = round(attack_size / 100 * n_genuine_users)

    # Compute filler_size as mean genuine profile length if not given
    if filler_size is None:
        filler_size = max(1, round(trn_mat.nnz / n_genuine_users))

    # --- Select target items (deterministic, from long tail) ---
    sorted_by_pop = np.argsort(item_popularity)          # ascending
    long_tail_end = max(num_targets, int(0.2 * n_items))
    long_tail      = sorted_by_pop[:long_tail_end]

    target_rng   = np.random.default_rng(target_seed)
    target_items = target_rng.choice(long_tail, size=num_targets, replace=False).tolist()
    target_set   = set(target_items)

    # If no fake users (attack_size=0), return original matrix unchanged
    if n_fake == 0:
        return trn_mat.tocoo(), target_items, 0

    # --- Build filler pool ---
    non_target = np.array([i for i in range(n_items) if i not in target_set])

    if strategy == 'bandwagon':
        pop_order  = np.argsort(item_popularity[non_target])[::-1]
        pool_size  = min(len(non_target), max(filler_size * 10, 200))
        filler_pool = non_target[pop_order[:pool_size]]
    else:  # random
        filler_pool = non_target

    actual_filler = min(filler_size, len(filler_pool))

    # --- Build one CSR row per fake user ---
    filler_rng = np.random.default_rng(target_seed + 1)
    rows_data  = []
    for _ in range(n_fake):
        chosen_filler = filler_rng.choice(filler_pool, size=actual_filler, replace=False)
        items = np.concatenate([np.array(target_items), chosen_filler])
        row   = np.zeros(n_items, dtype=np.float32)
        row[items] = 1.0
        rows_data.append(row)

    fake_csr  = csr_matrix(np.vstack(rows_data), dtype=np.float32)
    poisoned  = vstack([trn_mat.tocsr(), fake_csr], format='coo')

    return poisoned, target_items, n_fake


def make_fake_embeddings(genuine_user_emb, n_fake, mode='clone', seed=42):
    """
    Generate LLM embeddings for fake users.

    mode='clone' : copy a random genuine user's embedding (realistic).
    mode='mean'  : every fake user gets the mean of all genuine embeddings.

    Returns numpy array of shape (n_fake, emb_dim).
    """
    genuine = np.array(genuine_user_emb, dtype=np.float32)
    if mode == 'mean':
        mean_emb = genuine.mean(axis=0, keepdims=True)
        return np.repeat(mean_emb, n_fake, axis=0)
    else:  # clone
        rng     = np.random.default_rng(seed)
        indices = rng.integers(0, len(genuine), size=n_fake)
        return genuine[indices].copy()
