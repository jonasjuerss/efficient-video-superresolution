import wandb


def init_wandb(args):
    if args.use_wandb:
        wandb.init(project="efficient-superresolution", entity="camb-mphil", config=args)
        return wandb.config
    return args


def log(*args, **kwargs):
    if wandb.run is not None:
        wandb.log(*args, **kwargs)
