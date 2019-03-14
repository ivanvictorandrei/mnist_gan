import cv2
import torch
import numpy as np
from torch import nn, optim
import tensorflow as tf

from dataset import get_dataset, images_to_vectors, vectors_to_images
from models import noise, DiscriminatorNet, GeneratorNetwork


def ones_target(size):
    data = torch.Tensor(torch.ones(size, 1))
    return data


def zeros_target(size):
    data = torch.Tensor(torch.zeros(size, 1))
    return data


def train_discriminator(discriminator, loss, optimizer, real_data, fake_data):
    N = real_data.size(0)

    optimizer.zero_grad()

    prediction_real = discriminator(real_data)

    error_real = loss(prediction_real, ones_target(N))
    error_real.backward()

    prediction_fake = discriminator(fake_data)

    error_fake = loss(prediction_fake, zeros_target(N))
    error_fake.backward()

    optimizer.step()

    return error_real + error_fake, prediction_real, prediction_fake


def train_generator(discriminator, loss, optimizer, fake_data):
    N = fake_data.size(0)

    optimizer.zero_grad()

    prediction = discriminator(fake_data)

    error = loss(prediction, ones_target(N))
    error.backward()

    optimizer.step()

    return error


dataloader, num_batches = get_dataset()

discriminator = DiscriminatorNet()
generator = GeneratorNetwork()

d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

loss = nn.BCELoss()


num_test_samples = 16
test_noise = noise(num_test_samples)


summary_writer = tf.summary.FileWriter('./events')

g_error_summary = tf.placeholder(tf.float32, shape=())
d_error_summary = tf.placeholder(tf.float32, shape=())
images_summary = tf.placeholder(tf.uint8, shape=(None, 28, 28, 1))

tf.summary.scalar('Generator Error', g_error_summary)
tf.summary.scalar('Discriminator Error', d_error_summary)
tf.summary.image('Fake Images', images_summary)

merged_summaries = tf.summary.merge_all()


num_epochs = 200

with tf.Session(config=tf.ConfigProto(device_count={'CPU': 1, 'GPU': 0})) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(num_epochs):
        for n_batch, (real_batch, _) in enumerate(dataloader):
            N = real_batch.size(0)

            # Train discriminator
            real_data = torch.Tensor(images_to_vectors(real_batch))

            fake_data = generator(noise(N)).detach()

            d_error, d_pred_real, d_pred_fake = train_discriminator(
                discriminator, loss, d_optimizer, real_data, fake_data
            )


            # Train generator
            fake_data = generator(noise(N))

            g_error = train_generator(discriminator, loss, g_optimizer, fake_data)


            if n_batch % 100 == 0:

                test_images = vectors_to_images(generator(test_noise))
                test_images = test_images.detach().numpy()
                test_images_normalized = \
                cv2.normalize(test_images, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).\
                    astype(np.uint8)

                print("[{}, {}] - Discriminator error: {} - Generator error: {}".\
                      format(epoch, n_batch, d_error, g_error))

                summary, *res = sess.run(
                    [
                        merged_summaries,
                        g_error_summary,
                        d_error_summary,
                        images_summary
                    ],
                    feed_dict={
                        g_error_summary: g_error.item(),
                        d_error_summary: d_error.item(),
                        images_summary: test_images_normalized
                    }
                )

                summary_writer.add_summary(summary, global_step=epoch*num_batches+n_batch)