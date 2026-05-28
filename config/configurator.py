import os
import yaml
import pickle
import argparse

_ABLATION_PRESETS = {
    # Disable Adaptive Graph Structure learning (cf_index=None) + Info Bottleneck (beta=0)
    'wo_ags_ib': {'cf_index': None, 'model_patch': {'beta': 0.0}},
    # Disable preference Knowledge Distillation
    'wo_kd':     {'model_patch': {'prf_weight': 0.0}},
    # Disable all Semantic Embedding losses (alpha=0 zeroes loss_llm term)
    'wo_se':     {'model_patch': {'alpha': 0.0}},
}


def parse_configure(model=None, dataset=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lightgcn_agr', help='Model name')
    parser.add_argument('--dataset', type=str, default='amazon', help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--seed', type=int, default=2025, help='Random number')
    parser.add_argument('--cuda', type=str, default='0', help='Device number')
    parser.add_argument('--diverse', type=int, default=2, help='Diverse profile number')
    parser.add_argument('--ablation', type=str, default='none',
                        help='Ablation variant: none|wo_ags_ib|wo_kd|wo_se')
    parser.add_argument('--attack_config', type=str, default=None,
                        help='Path to YAML file with an attack: section to merge into configs')
    args, _ = parser.parse_known_args()

    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    model_name = model.lower() if model else args.model.lower() if args.model else 'default'
    if dataset:
        args.dataset = dataset

    pre_dir = os.getcwd()
    config_path = f"{pre_dir}/config/models_config/{model_name}.yml"
    if not os.path.exists(config_path):
        raise Exception("Please create the yaml file for your model first.")

    with open(config_path, encoding='utf-8') as f:
        configs = yaml.safe_load(f)
    configs['model']['name'] = configs['model']['name'].lower()
    configs.setdefault('tune', {'enable': False})
    configs['device'] = args.device
    configs['diverse'] = args.diverse
    if args.dataset:
        configs['data']['name'] = args.dataset
    if args.seed is not None:
        configs['train']['seed'] = args.seed

    # Merge optional attack config YAML
    if args.attack_config and os.path.exists(args.attack_config):
        with open(args.attack_config, encoding='utf-8') as f:
            attack_extra = yaml.safe_load(f)
        if attack_extra:
            configs.update(attack_extra)

    # Apply ablation preset overrides
    ablation = args.ablation.lower() if args.ablation else 'none'
    if ablation in _ABLATION_PRESETS:
        preset = _ABLATION_PRESETS[ablation]
        if 'cf_index' in preset:
            configs['cf_index'] = preset['cf_index']
        if 'model_patch' in preset:
            dataset_name = configs['data']['name']
            for k, v in preset['model_patch'].items():
                configs['model'][k] = v
                if dataset_name in configs['model'] and isinstance(configs['model'][dataset_name], dict):
                    configs['model'][dataset_name][k] = v

    user_embedding_path = f"{pre_dir}/data/{configs['data']['name']}/usr_emb_np.pkl"
    item_embedding_path = f"{pre_dir}/data/{configs['data']['name']}/itm_emb_np.pkl"
    with open(user_embedding_path, 'rb') as f:
        configs['user_embedding'] = pickle.load(f)
    with open(item_embedding_path, 'rb') as f:
        configs['item_embedding'] = pickle.load(f)

    # for index in range(configs['diverse'] - 2):
    #     user_embedding_index_path = f"{pre_dir}/data/{configs['data']['name']}/diverse_profile/diverse_user_embedding_{index + 1}.pkl"
    #     item_embedding_index_path = f"{pre_dir}/data/{configs['data']['name']}/diverse_profile/diverse_item_embedding_{index + 1}.pkl"
    #     with open(user_embedding_index_path, 'rb') as f:
    #         configs[f'user_embedding_{index + 1}'] = pickle.load(f)
    #     with open(item_embedding_index_path, 'rb') as f:
    #         configs[f'item_embedding_{index + 1}'] = pickle.load(f)

    return configs

configs = parse_configure()