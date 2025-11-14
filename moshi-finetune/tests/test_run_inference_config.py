import json
from pathlib import Path


def compute_extended_context_from_config(cfg: dict) -> int:
    original_context = cfg.get('context', 3000)
    yarn_cfg = cfg.get('yarn_config', {})
    if yarn_cfg.get('enabled', False):
        scale = yarn_cfg.get('scale', 1.0)
        return int(original_context * scale)
    return original_context


def test_extended_context_computation_from_config_file():
    # Use an existing checkpoint config if present in the repo, otherwise create a synthetic config
    ckpt_cfg_path = Path(__file__).parent.parent / 'ttt_training_run2' / 'checkpoints' / 'checkpoint_002000' / 'consolidated' / 'config.json'
    if ckpt_cfg_path.exists():
        cfg = json.loads(ckpt_cfg_path.read_text())
    else:
        cfg = {
            'context': 3000,
            'yarn_config': {'enabled': True, 'scale': 4.0, 'original_max_seq_len': 3000}
        }

    extended = compute_extended_context_from_config(cfg)
    assert isinstance(extended, int)
    if cfg.get('yarn_config', {}).get('enabled', False):
        assert extended == int(cfg.get('context', 3000) * cfg['yarn_config']['scale'])
    else:
        assert extended == cfg.get('context', 3000)
