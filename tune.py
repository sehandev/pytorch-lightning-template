# Standard

# PIP
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.integration.pytorch_lightning import TuneReportCallback

# Custom
from config import Config
from c_dataset import CustomDataModule
from module import CustomModule


ray.init(dashboard_host='0.0.0.0')


def train_tune(config, num_gpus=0, num_epochs=10):
    cfg = Config(config['seed'])

    wandb_logger = WandbLogger(
        project=f'tune_{cfg.PROJECT_TITLE}',
        name=cfg.WANDB_NAME,
    )

    tune_report_callback = TuneReportCallback(
        {
            'loss': 'val_loss',
        },
        on='validation_end',
    )

    trainer = Trainer(
        gpus=num_gpus,
        max_epochs=num_epochs,
        logger=wandb_logger,
        progress_bar_refresh_rate=0,
        accelerator='dp',
        deterministic=True,
        precision=16,
        callbacks=[
            tune_report_callback,
        ],
    )

    data_module = CustomDataModule(
        seq_len=cfg.SEQ_LEN,
        batch_size=config['batch_size'],
        num_workers=cfg.NUM_WORKERS,
    )

    model = CustomModule(
        model_option=cfg.model_option,
        max_epochs=cfg.MAX_EPOCHS,
        learning_rate=config['learning_rate'],
        criterion_name=cfg.CRITERION,
        optimizer_name=cfg.OPTIMIZER,
        lr_scheduler_name=cfg.LR_SCHEDULER,
    )

    trainer.fit(model, datamodule=data_module)


def tune_asha(num_samples=10, num_epochs=10):
    num_cpus = 2
    num_gpus = 1

    config = {
        'seed': tune.randint(0, 1000),
        'learning_rate': tune.loguniform(1e-6, 1e-2),
        'batch_size': tune.choice([128, 256, 512]),
    }

    optuna_search = OptunaSearch(
        metric="loss",
        mode="min",
    )

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=20,
        reduction_factor=2,
    )

    reporter = CLIReporter(
        parameter_columns=['seed', 'learning_rate', 'batch_size'],
        metric_columns=['loss', 'training_iteration'],
        max_progress_rows=30,
        sort_by_metric=True,
    )

    analysis = tune.run(
        tune.with_parameters(
            train_tune,
            num_gpus=num_gpus,
            num_epochs=num_epochs,
        ),
        resources_per_trial={
            'cpu': num_cpus,
            'gpu': num_gpus,
        },
        metric='loss',
        mode='min',
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        search_alg=optuna_search,
        progress_reporter=reporter,
        name='tune',
        # resume=True,
    )

    print('Best hyperparameters found were: ', analysis.best_config)


tune_asha(
    num_samples=30,
    num_epochs=50,
)
