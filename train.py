import os, glob, random, traceback
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MAE
from tensorflow.math import reduce_mean
from models.gan import Generator, MultiScaleDiscriminator
from util.hparams import *


data_dir = './data'
mel_list = sorted(glob.glob(os.path.join(data_dir + '/mel', '*.npy')))
audio_list = sorted(glob.glob(os.path.join(data_dir + '/audio', '*.npy')))


def DataGenerator():
    while True:
        idx_list = np.random.choice(len(mel_list), batch_size ** 3, replace=False)
        random.shuffle(idx_list)
        idx_list = [idx_list[i : i + batch_size] for i in range(0, len(idx_list), batch_size)]
        
        for idx in idx_list:
            mel, audio = [], []
            for i in idx:
                mel_npy = np.load(mel_list[i])
                mel_start = random.randint(0, mel_npy.shape[0] - seq_len - 1)
                mel_npy = mel_npy[mel_start : mel_start + seq_len, :]
                mel.append(mel_npy)
                
                audio_npy = np.load(audio_list[i])
                audio_start = mel_start * hop_length
                audio_npy = audio_npy[audio_start : audio_start + seq_len * hop_length]
                audio_npy = np.expand_dims(audio_npy, axis=-1)
                audio.append(audio_npy)

            yield mel, audio


@tf.function()
def train_step(mel, audio):
    with tf.GradientTape() as d_tape:
        d_real = d(audio)
        d_loss_real = 0
        for scale in d_real:
            d_loss_real += reduce_mean(tf.nn.relu(1. - scale[-1]))

        fake_audio = g(mel)
        d_fake = d(fake_audio)
        d_loss_fake = 0
        for scale in d_fake:
            d_loss_fake += reduce_mean(tf.nn.relu(1. + scale[-1]))

        d_loss = d_loss_real + d_loss_fake

    d_variables = d.trainable_variables
    d_gradients = d_tape.gradient(d_loss, d_variables)
    d_opt.apply_gradients(zip(d_gradients, d_variables))

    with tf.GradientTape() as g_tape:
        d_fake = d(g(mel))
        g_loss = 0
        for scale in d_fake:
            g_loss += reduce_mean(-scale[-1])

        feature_loss = 0
        for i in range(3):
            for j in range(len(d_fake[i])-1):
                feature_loss += reduce_mean(MAE(d_fake[i][j], d_real[i][j]))

        g_loss += lambda_feat * feature_loss

    g_variables = g.trainable_variables
    g_gradients = g_tape.gradient(g_loss, g_variables)
    g_opt.apply_gradients(zip(g_gradients, g_variables))
    
    return d_loss, g_loss


dataset = tf.data.Dataset.from_generator(generator=DataGenerator,
                                         output_types=(tf.float32, tf.float32),
                                         output_shapes=(tf.TensorShape([batch_size, None, mel_dim]),
                                                        tf.TensorShape([batch_size, None, 1])))\
    .prefetch(tf.data.experimental.AUTOTUNE)

g = Generator()
d = MultiScaleDiscriminator()
g_opt = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
d_opt = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
step = tf.Variable(0)

os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint = tf.train.Checkpoint(g_opt=g_opt, d_opt=d_opt, g=g, d=d, step=step)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=100)

checkpoint.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print('Restore checkpoint from {}'.format(manager.latest_checkpoint))

try:
    for mel, audio in dataset:
        d_loss, g_loss = train_step(mel, audio)
        checkpoint.step.assign_add(1)
        print("Step: {}, D_Loss: {:.5f}, G_Loss: {:.5f}".format(int(checkpoint.step), d_loss, g_loss))

        if int(checkpoint.step) % checkpoint_step == 0:
            checkpoint.save(file_prefix=os.path.join(checkpoint_dir, 'ckpt-{}'.format(int(checkpoint.step))))

except Exception:
    traceback.print_exc()