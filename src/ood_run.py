import os, shutil
from typing import Tuple, Optional

import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import open_dict, DictConfig
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.utils.data import dataset_info, monitor_dict
from src.utils.logging import get_logger
from src.utils.callbacks import BestPerformance
from src.utils.expl import attr_algos, baseline_required


def get_callbacks(cfg: DictConfig):

    monitor = monitor_dict[cfg.data.dataset]
    mode = cfg.data.mode
    callbacks = [
        BestPerformance(monitor=monitor, mode=mode)
    ]

    if cfg.save_checkpoint:
        callbacks.append(
            ModelCheckpoint(
                monitor=monitor,
                dirpath=os.path.join(cfg.save_dir, 'checkpoints'),
                save_top_k=1,
                mode=mode,
                verbose=True,
                save_last=False,
                save_weights_only=True,
            )
        )

    if cfg.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                min_delta=0.00,
                patience=cfg.training.patience,
                verbose=False,
                mode=mode
            )
        )

    return callbacks


logger = get_logger(__name__)


def build(cfg) -> Tuple[pl.LightningDataModule, pl.LightningModule, pl.Trainer]:
    dm = instantiate(
        cfg.data,
        attr_algo=cfg.model.attr_algo,
        train_shuffle=cfg.training.train_shuffle,
    )
    dm.setup(splits=cfg.training.eval_splits.split(","))

    logger.info(f'load {cfg.data.dataset} <{cfg.data._target_}>')

    model = instantiate(
        cfg.model, num_classes=dataset_info[cfg.data.dataset]['num_classes'],
        _recursive_=False
    )
    logger.info(f'load {cfg.model.arch} <{cfg.model._target_}>')

    run_logger = instantiate(cfg.logger, cfg=cfg, _recursive_=False)

    with open_dict(cfg):
        if cfg.debug or cfg.logger.offline:
            exp_dir = cfg.logger.name
            cfg.logger.neptune_exp_id = cfg.logger.name
        else:
            if cfg.logger.logger == "neptune":
                exp_dir = run_logger.experiment_id
                cfg.logger.neptune_exp_id = run_logger.experiment_id
            else:
                raise NotImplementedError
        cfg.save_dir = os.path.join(cfg.save_dir, exp_dir)
        os.makedirs(cfg.save_dir, exist_ok=True)

        # copy hydra configs
        shutil.copytree(
            os.path.join(os.getcwd(), ".hydra"),
            os.path.join(cfg.save_dir, "hydra")
        )

    logger.info(f"saving to {cfg.save_dir}")

    trainer = instantiate(
        cfg.trainer,
        callbacks=get_callbacks(cfg),
        checkpoint_callback=cfg.save_checkpoint,
        logger=run_logger,
        _convert_="all",
    )

    return dm, model, trainer


def restore_config_params(model, cfg: DictConfig):
    for key, val in cfg.model.items():
        if hasattr(model, key):
            setattr(model, key, val)
        else:
            setattr(model, key, val)

    if cfg.model.save_outputs:
        assert cfg.model.exp_id in cfg.training.ckpt_path

    if (cfg.model.expl_reg and cfg.model.explainer_type == 'attr_algo'):
        model.attr_func = attr_algos[model.attr_algo](model)
        model.tokenizer = AutoTokenizer.from_pretrained(cfg.model.arch)
        model.baseline_required = baseline_required[model.attr_algo]
        model.word_emb_layer = model.task_encoder.embeddings.word_embeddings

    logger.info('Restored params from model config.')

    return model


def run(cfg: DictConfig) -> Optional[float]:
    pl.seed_everything(cfg.seed)
    dm, model, trainer = build(cfg)
    pl.seed_everything(cfg.seed)

    if cfg.save_rand_checkpoint:
        ckpt_path = os.path.join(cfg.save_dir, 'checkpoints', 'rand.ckpt')
        logger.info(f"Saving randomly initialized model to {ckpt_path}")
        trainer.model = model
        trainer.save_checkpoint(ckpt_path)
    elif not cfg.training.evaluate_ckpt:
        # either train from scratch, or resume training from ckpt
        if cfg.training.finetune_ckpt:
            assert cfg.training.ckpt_path
            save_dir = '/'.join(cfg.save_dir.split('/')[:-1])
            ckpt_path = os.path.join(save_dir, cfg.training.ckpt_path)
            model = model.load_from_checkpoint(ckpt_path, strict=False)
            model = restore_config_params(model, cfg)
            logger.info(f"Loaded checkpoint (for fine-tuning) from {ckpt_path}")

            if cfg.finetune_heads:
                for modoule in [model.expl_encoder, model.task_encoder]:
                    for n, p in module.named_parameters():
                        p.requires_grad = False

        trainer.fit(model=model, datamodule=dm)

        if getattr(cfg, "tune_metric", None):
            metric = trainer.callback_metrics[cfg.tune_metric].detach()
            logger.info(f"best metric {metric}")
            return metric
    else:
        # evaluate the pretrained model on the provided splits
        assert cfg.training.ckpt_path
        save_dir = '/'.join(cfg.save_dir.split('/')[:-2])
        ckpt_path = os.path.join(save_dir, cfg.training.ckpt_path)
        model = model.load_from_checkpoint(ckpt_path, strict=False)
        logger.info(f"Loaded checkpoint for evaluation from {cfg.training.ckpt_path}")
        model = restore_config_params(model, cfg)

        # if cfg.ood:
        model.max_length = dataset_info[cfg.model.dataset]['max_length'][cfg.model.arch]
        # if cfg.model.compute_attr:
        #     model.attr_dict = {
        #         'attr_algo': cfg.model.attr_algo,
        #         'baseline_required': baseline_required[cfg.model.attr_algo],
        #         'attr_func': attr_algos[cfg.model.attr_algo](model),
        #         'tokenizer': AutoTokenizer.from_pretrained(model.arch),
        #     }
        print('Evaluating loaded model checkpoint...')
        for split in cfg.training.eval_splits.split(','):
            print(f'Evaluating on split: {split}')
            if split == 'train':
                loader = dm.train_dataloader()
            elif split == 'dev':
                loader = dm.val_dataloader(test=True)
            elif split == 'test':
                loader = dm.test_dataloader()
            trainer.test(model=model, dataloaders=loader)
