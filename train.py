import argparse
import tensorflow as tf

import wandb

from dataloader import FrameGenerator
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

    output_signature = (tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32))

    train_ds = tf.data.Dataset.from_generator(FrameGenerator("HR", "tecogan_generate", training=True),
                                              output_signature=output_signature)
    val_ds = tf.data.Dataset.from_generator(FrameGenerator("HR", "tecogan_generate", training=True),
                                            output_signature=output_signature)
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

    for a in train_ds.take(10):
        print(*a)

    train_frames, train_labels = next(iter(train_ds))
    print(f'Shape of training set of frames: {train_frames.shape}')
    print(f'Shape of training labels: {train_labels.shape}')

    val_frames, val_labels = next(iter(val_ds))
    print(f'Shape of validation set of frames: {val_frames.shape}')
    print(f'Shape of validation labels: {val_labels.shape}')

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_ds,
              epochs=10,
              validation_data=val_ds,
              callbacks=tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'))