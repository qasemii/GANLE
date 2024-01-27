import time

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # import here for faster auto completion
    from src.utils.conf import touch

    if cfg.model.dataset =='allennlp':
        cfg.model.dataset = 'mnli'
        from src.allennlp_run import run
    elif cfg.model.dataset in ['mnli', 'hans', 'emnli', 'mnli_contrast_original', 'mnli_contrast_contrast']:
        from src.ood_run import run
        touch(cfg)
    else:
        from src.run import run
        touch(cfg)

    # additional set field by condition
    # assert no missing etc

    start_time = time.time()
    metric = run(cfg)
    print(
        f'Time Taken for experiment {cfg.logger.neptune_exp_id}: {(time.time() - start_time) / 3600}h')

    return metric


if __name__ == '__main__':
    main()
