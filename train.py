import argparse
import tensorflow as tf

import wandb

from wandb_utils import init_wandb

parser = argparse.ArgumentParser(description='DNN curve training')
parser.add_argument('--seed', type=int, default=1,
                    help='The seed.')
parser.set_defaults(use_wandb=True)
parser.add_argument('--no_wandb', action='store_false', dest='use_wandb',
                    help='Turns off logging to wandb')



if __name__ == "__main__":
    args = parser.parse_args()
    init_wandb(args)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=args.seed,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=args.seed,
        image_size=(img_height, img_width),
        batch_size=batch_size)