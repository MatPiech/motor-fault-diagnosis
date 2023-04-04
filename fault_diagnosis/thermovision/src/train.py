from typing import List, Optional

import numpy as np
import hydra
import onnx
import torch
from omegaconf import DictConfig
from onnxsim import simplify
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import LightningLoggerBase

from src.utils import utils, eval_utils

log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if (seed := config.get('seed')) is not None:
        seed_everything(seed, workers=True)

    # Init lightning datamodule
    log.info(f'Instantiating datamodule <{config.datamodule._target_}>')

    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule, seed=config.get('seed', 42))

    # Init lightning model
    log.info(f'Instantiating model <{config.model._target_}>')
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if 'callbacks' in config:
        for _, cb_conf in config.callbacks.items():
            if '_target_' in cb_conf:
                log.info(f'Instantiating callback <{cb_conf._target_}>')
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if 'logger' in config:
        for _, lg_conf in config.logger.items():
            if '_target_' in lg_conf:
                log.info(f'Instantiating logger <{lg_conf._target_}>')
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f'Instantiating trainer <{config.trainer._target_}>')
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_='partial'
    )

    # Send some parameters from config to all lightning loggers
    log.info('Logging hyperparameters!')
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    if not config.eval_mode:
        log.info('Starting training!')
        trainer.fit(model=model, datamodule=datamodule)

    # Evaluate model on test set, using the best model achieved during training
    if config.get('test_after_training'):
        log.info('Starting testing!')
        if config.eval_mode:
            trainer.test(model=model, datamodule=datamodule, ckpt_path=config.trainer.resume_from_checkpoint)
            outputs = trainer.predict(model=model, datamodule=datamodule, ckpt_path=config.trainer.resume_from_checkpoint)
        else:
            trainer.test(model=model, datamodule=datamodule, ckpt_path='best')
            outputs = trainer.predict(model=model, datamodule=datamodule, ckpt_path='best')

        class_mapping = config.model.classes
        y_true = []
        y_pred = []

        for (gt, pred) in outputs:
            pred = torch.argmax(pred, dim=1)

            y_true.append(gt.numpy())
            y_pred.append(pred.numpy())

        y_true = [class_mapping[idx] for idx in np.hstack(y_true).tolist()]
        y_pred = [class_mapping[idx] for idx in np.hstack(y_pred).tolist()]

        eval_utils.calculate_metrics_per_class(y_true, y_pred, class_mapping)

    # Make sure everything closed properly
    log.info('Finalizing!')
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    if not config.eval_mode:
        log.info(f'Best model ckpt: {trainer.checkpoint_callback.best_model_path}')

    if config.get('export').get('export_to_onnx'):
        opset = config.get('export').get('opset')
        use_simplifier = config.get('export').get('use_simplifier')
        log.info(f'Export model to onnx! Params: opset={opset}, use_simplifier={use_simplifier}')

        model.eval()
        x = next(iter(datamodule.test_dataloader()))[0][:1]

        torch.onnx.export(model.network,
                          x,  # model input (or a tuple for multiple inputs)
                          'model.onnx',  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=opset,  # the ONNX version to export the model to
                          input_names=['input'],
                          output_names=['output'],
                          do_constant_folding=False)
        
        if use_simplifier:
            model = onnx.load('model.onnx')
            model_simp, check = simplify(model)
            assert check, 'Simplified ONNX model could not be validated'
            onnx.save(model_simp, 'model.onnx')

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get('optimized_metric')
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]
