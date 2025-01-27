from lightning import LightningModule, Trainer
from lightning.pytorch.cli import ArgsType, LightningCLI, SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger
import torch
import json
import src as src
from dotenv import load_dotenv
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch import seed_everything

seed_everything(123, workers=True)
load_dotenv()
# torch.set_float32_matmul_precision('medium')

class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if isinstance(trainer.logger, WandbLogger):
            config = self.parser.dump(self.config, skip_none=False, format="json")  # Required for proper reproducibility
            trainer.logger.log_hyperparams(json.loads(config))

# class MyLightningCLI(LightningCLI):
#     def add_arguments_to_parser(self, parser):
#         parser.add_argument("--paths", type=dict, default="{}")


def lightning_cli_run(args: ArgsType = None):
    
    cli = LightningCLI(
        save_config_callback=LoggerSaveConfigCallback,
        save_config_overwrite=True,
        seed_everything_default=123,
        # save_config_kwargs={"save_to_log_dir": False},
        # auto_configure_optimizers=False,
        parser_kwargs={
                        "parser_mode": "omegaconf", 
                        "fit": {"default_config_files": ["configs/grnformer.yaml"]}
                        },
        args=args
    )

if __name__ == "__main__":
    lightning_cli_run()
