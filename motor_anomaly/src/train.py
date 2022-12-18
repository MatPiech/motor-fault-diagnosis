from typing import List, Optional

import numpy as np
from sklearn import metrics
from tabulate import tabulate
import hydra
import onnx
import timm.data
import torch
from omegaconf import DictConfig
from onnxsim import simplify
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import LightningLoggerBase

from src.utils import utils

log = utils.get_logger(__name__)


def print_metrics(y_true, y_pred, class_mapping):
    total_predictions = [0, 0, 0]
    total_true = [0, 0, 0]

    cm = metrics.confusion_matrix(y_true, y_pred, labels=class_mapping)

    for i in range(3):
        for j in range(3):
            total_predictions[i] += cm[j][i]
            total_true[i] += cm[i][j]

    # recall and precision for each class
    r_h = cm[0][0] / total_predictions[0]
    r_m = cm[1][1] / total_predictions[1]
    r_r = cm[2][2] / total_predictions[2]
    p_h = cm[0][0] / total_true[0]
    p_m = cm[1][1] / total_true[1]
    p_r = cm[2][2] / total_true[2]

    # data for confusion matrix
    data = [["", class_mapping[0], class_mapping[1], class_mapping[2], "total"],
            [class_mapping[0], cm[0][0], cm[0][1], cm[0][2], total_true[0]],
            [class_mapping[1], cm[1][0], cm[1][1], cm[1][2], total_true[1]],
            [class_mapping[2], cm[2][0], cm[2][1], cm[2][2], total_true[2]],
            ["Total predicted:", total_predictions[0], total_predictions[1], total_predictions[2], sum(total_true)]]

    metrics_data = [
        ["", class_mapping[0], class_mapping[1], class_mapping[2],],
        ["F1 Score", round((2 * p_h * r_h) / (p_h + r_h), 2), round((2 * p_m * r_m) / (p_m + r_m), 2), round((2 * p_r * r_r) / (p_r + r_r), 2)],
        ["Recall", round(r_h, 2), round(r_m, 2), round(r_r, 2)],
        ["Precision", round(p_h, 2), round(p_m, 2), round(p_r, 2)]
    ]

    # printing confusion matrix, f1 score, recall and precision
    print(tabulate(data, tablefmt="simple_grid"))
    print(tabulate(metrics_data, tablefmt="simple_grid"))


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
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule,
                                                              image_mean=timm.data.IMAGENET_DEFAULT_MEAN,
                                                              image_std=timm.data.IMAGENET_DEFAULT_STD)

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
    log.info('Starting training!')
    if not config.eval_mode:
        trainer.fit(model=model, datamodule=datamodule)

    # Evaluate model on test set, using the best model achieved during training
    if config.get('test_after_training'):
        log.info('Starting testing!')
        if config.eval_mode:
            trainer.test(model=model, datamodule=datamodule, ckpt_path=config.trainer.resume_from_checkpoint)
        else:
            trainer.test(model=model, datamodule=datamodule, ckpt_path='best')

        class_mapping = ['healthy', 'misalignment', 'broken rotor']
        y_true = []
        y_pred = []

        outputs = trainer.predict(model=model, datamodule=datamodule, ckpt_path='best')
        for o in outputs:
            gt, pred = o
            pred = torch.argmax(pred, dim=1)

            y_true.append(gt.numpy())
            y_pred.append(pred.numpy())

        y_true = [class_mapping[idx] for idx in np.hstack(y_true).tolist()]
        y_pred = [class_mapping[idx] for idx in np.hstack(y_pred).tolist()]

        print_metrics(y_true, y_pred, class_mapping)

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
        x = next(iter(datamodule.test_dataloader()))[0]

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
