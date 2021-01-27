import numpy as np
import tensorflow as tf
import time
from IPython import display
import datetime
import os
import logging

import model

tf.keras.backend.set_floatx('float32')

@tf.function
def tracegraph(x, model):
    return model(x)

def run_fit(gen, dis, dataset, epochs, len_high_size,
        scale, valid_dataset=None, log_dir=None, saved_model_dir=None):
    if log_dir is None:
        log_dir = './logs/model'
    logging.basicConfig(filename=os.path.join(
        log_dir, 'training.log'), level=logging.INFO)
    if saved_model_dir is None:
        saved_model_dir = './saved_model'
    generator_optimizer_low = tf.keras.optimizers.Adam()
    generator_optimizer_high = tf.keras.optimizers.Adam()
    discriminator_optimizer = tf.keras.optimizers.Adam()
    opts = [generator_optimizer_low, generator_optimizer_high]

    # for generator#, discriminator_optimizer]
    generator_log_ssim_low = tf.keras.metrics.Mean(
        'train_gen_low_ssim_loss', dtype=tf.float32)
    generator_log_mse_low = tf.keras.metrics.Mean(
        'train_gen_low_mse_loss', dtype=tf.float32)
    generator_log_mse_high = tf.keras.metrics.Mean(
        'train_gen_high_mse_loss', dtype=tf.float32)
    generator_log_bce_high = tf.keras.metrics.Mean(
        'train_gen_high_bce_loss', dtype=tf.float32)
    generator_log_ssim_high = tf.keras.metrics.Mean(
        'train_gen_high_ssim_loss', dtype=tf.float32)
    discriminator_log = tf.keras.metrics.Mean(
        'train_discriminator_loss', dtype=tf.float32)
    if valid_dataset is not None:
        valid_gen_log_h_bce = tf.keras.metrics.Mean(
            'valid_gen_high_bce_loss', dtype=tf.float32)
        valid_gen_log_h_mse = tf.keras.metrics.Mean(
            'valid_gen_high_mse_loss', dtype=tf.float32)
        valid_gen_log_h_ssim = tf.keras.metrics.Mean(
            'valid_gen_high_ssim_loss', dtype=tf.float32)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    train_log_dir = os.path.join(log_dir, current_time, 'generator')
    train_summary_G_writer = tf.summary.create_file_writer(train_log_dir)
    train_log_dir = os.path.join(log_dir, current_time, 'discriminator')
    train_summary_D_writer = tf.summary.create_file_writer(train_log_dir)

    demo_log_dir = os.path.join(log_dir, current_time, 'valid')
    demo_writer = tf.summary.create_file_writer(demo_log_dir)

    train_log_dir = os.path.join(log_dir, current_time, 'model')
    writer = tf.summary.create_file_writer(train_log_dir)
    tf.summary.trace_on(graph=True, profiler=False)
    # Forward pass
    tracegraph(tf.zeros((1, len_high_size, len_high_size, 1)), gen)
    with writer.as_default():
        tf.summary.trace_export(name="model_gen_trace",
                                step=0, profiler_outdir=train_log_dir)
    tf.summary.trace_on(graph=True, profiler=False)

    tracegraph(tf.zeros((1, len_high_size, len_high_size, 1)), dis)
    with writer.as_default():
        tf.summary.trace_export(name="model_dis_trace",
                                step=0, profiler_outdir=train_log_dir)

    with demo_writer.as_default():
        [_, (demo_input_low, demo_input_high)] = next(enumerate(dataset.take(1)))
        mpy = demo_input_low.numpy()
        m = np.log1p(1000*np.squeeze(mpy[:, :, :, 0]))
        fig = plot_matrix(m)
        images = plot_to_image(fig)
        tf.summary.image("demo data low examples",
                         images, max_outputs=16, step=0)
        mpy = demo_input_high.numpy()
        m = np.log1p(1000*np.squeeze(mpy[:, :, :, 0]))
        fig = plot_matrix(m)
        images = plot_to_image(fig)
        tf.summary.image("demo data high examples",
                         images, max_outputs=16, step=0)

    len_x2 = int(len_high_size/2)
    len_x4 = int(len_high_size/4)
    loss_filter_low_x2 = np.ones(shape=(len_x2, len_x2)) - \
        np.diag(np.ones(shape=(len_x2,)), k=0) - \
        np.diag(np.ones(shape=(len_x2-1,)), k=-1) - \
        np.diag(np.ones(shape=(len_x2-1,)), k=1)
    loss_filter_low_x4 = np.ones(shape=(len_x4, len_x4)) - \
        np.diag(np.ones(shape=(len_x4,)), k=0) - \
        np.diag(np.ones(shape=(len_x4-1,)), k=-1) - \
        np.diag(np.ones(shape=(len_x4-1,)), k=1)
    loss_filter_high = np.ones(shape=(len_high_size, len_high_size)) - \
        np.diag(np.ones(shape=(len_high_size,)), k=0) - \
        np.diag(np.ones(shape=(len_high_size-1,)), k=-1) - \
        np.diag(np.ones(shape=(len_high_size-1,)), k=1)

    [_, (demo_input_low, demo_input_high)] = next(enumerate(dataset.take(1)))

    train_step_generator = tf.function(model._train_step_generator)
    train_step_discriminator = tf.function(model._train_step_discriminator)
    best_loss = None
    for epoch in range(epochs):
        start = time.time()
        # train

        generator_log_ssim_low.reset_states()
        generator_log_mse_low.reset_states()
        generator_log_ssim_high.reset_states()
        generator_log_mse_high.reset_states()
        generator_log_bce_high.reset_states()

        discriminator_log.reset_states()

        if(epoch <= int(40)):
            loss_weights = [0.0, 10.0, 0.0]
        else:
            loss_weights = [0.1, 10.0, 0.0]

        for i, (low_m, high_m) in enumerate(dataset):

            g_ssim_l, g_mse_l, g_bce_h, g_mse_h, g_ssim_h = \
                train_step_generator(Gen=gen, Dis=dis,
                                        imgl=tf.dtypes.cast(
                                            low_m, tf.float32),
                                        imgr=tf.dtypes.cast(
                                            high_m, tf.float32),
                                        loss_filter=[loss_filter_low_x2, loss_filter_low_x4,
                                                    loss_filter_high],
                                        loss_weights=loss_weights,
                                        opts=opts)
            generator_log_ssim_low.update_state(g_ssim_l)
            generator_log_mse_low.update_state(g_mse_l)
            generator_log_ssim_high.update_state(g_ssim_h)
            generator_log_mse_high.update_state(g_mse_h)
            generator_log_bce_high.update_state(g_bce_h)

            # Gen, Dis, imgl, imgr, loss_filter, opts, train_logs
            d_loss = train_step_discriminator(Gen=gen, Dis=dis,
                                                imgl=tf.dtypes.cast(
                                                    low_m, tf.float32),
                                                imgr=tf.dtypes.cast(
                                                    high_m, tf.float32),
                                                loss_filter=[
                                                    loss_filter_high],
                                                opts=[discriminator_optimizer])
            discriminator_log.update_state(d_loss)

        # save model weights as checkpoints
        g_loss = g_bce_h*loss_weights[0] + g_mse_h*loss_weights[1]
        if best_loss is None or g_loss < best_loss:
            gen.save_weights(os.path.join(
                saved_model_dir, current_time, 'gen_weights_'+str(len_high_size)))
            dis.save_weights(os.path.join(
                saved_model_dir, current_time, 'dis_weights_'+str(len_high_size)))
            best_loss = g_loss

        # valid dataset
        if valid_dataset is not None:
            valid_gen_log_h_bce.reset_states()
            valid_gen_log_h_mse.reset_states()
            valid_gen_log_h_ssim.reset_states()
            for i, (low_m, high_m) in enumerate(valid_dataset):
                [dpl_x2, dpl_x4, dph, _, _] = gen(low_m, training=False)
                mfilter_high = tf.expand_dims(loss_filter_high, axis=0)
                mfilter_high = tf.expand_dims(mfilter_high, axis=-1)
                mfilter_high = tf.cast(mfilter_high, tf.float32)
                fake_hic_h = tf.multiply(dph, mfilter_high)
                imgr_filter = tf.multiply(high_m, mfilter_high)
                disc_generated_output = dis(fake_hic_h, training=False)

                valid_gen_log_h_bce.update_state(model.generator_bce_loss(disc_generated_output))
                valid_gen_log_h_mse.update_state(model.generator_mse_loss(fake_hic_h, imgr_filter))
                valid_gen_log_h_ssim.update_state(model.generator_ssim_loss(fake_hic_h, imgr_filter))


        [dpl_x2, dpl_x4, dph, _, _] = gen(demo_input_low, training=False)
        demo_disc_generated = dis(dph, training=False)
        demo_disc_true = dis(demo_input_high, training=False)

        with train_summary_G_writer.as_default():
            tf.summary.scalar('loss_gen_low_disssim',
                              generator_log_ssim_low.result(), step=epoch)
            tf.summary.scalar('loss_gen_low_mse',
                              generator_log_mse_low.result(), step=epoch)
            tf.summary.scalar('loss_gen_high_mse',
                              generator_log_mse_high.result(), step=epoch)
            tf.summary.scalar('loss_gen_high_disssim',
                              generator_log_ssim_high.result(), step=epoch)
            tf.summary.scalar('loss_gen_high_bce',
                              generator_log_bce_high.result(), step=epoch)
            if valid_dataset is not None:
                tf.summary.scalar('valid_gen_high_bce_loss',
                                  valid_gen_log_h_bce.result(), step=epoch)
                tf.summary.scalar('valid_gen_high_mse_loss',
                                  valid_gen_log_h_mse.result(), step=epoch)
                tf.summary.scalar('valid_gen_high_disssim_loss',
                                  valid_gen_log_h_ssim.result(), step=epoch)
            mpy = dpl_x2.numpy()
            m = np.log1p(1000*np.squeeze(mpy[:, :, :, 0]))
            fig = plot_matrix(m)
            image = plot_to_image(fig)
            tf.summary.image(name='gen_low_x2', data=image, step=epoch)
            mpy = dpl_x4.numpy()
            m = np.log1p(1000*np.squeeze(mpy[:, :, :, 0]))
            fig = plot_matrix(m)
            image = plot_to_image(fig)
            tf.summary.image(name='gen_low_x4', data=image, step=epoch)
            mpy = dph.numpy()
            m = np.log1p(1000*np.squeeze(mpy[:, :, :, 0]))
            fig = plot_matrix(m)
            image = plot_to_image(fig)
            tf.summary.image(name='gen_high', data=image, step=epoch)

        with train_summary_D_writer.as_default():
            tf.summary.scalar(
                'loss_dis', discriminator_log.result(), step=epoch)
            mpy = demo_disc_generated.numpy()
            m = np.squeeze(mpy).reshape((2, 2))
            fig = plot_prob_matrix(m)
            image = plot_to_image(fig)
            tf.summary.image(name='dis_gen', data=image, step=epoch)
            mpy = demo_disc_true.numpy()
            m = np.squeeze(mpy).reshape((2, 2))
            fig = plot_prob_matrix(m)
            image = plot_to_image(fig)
            tf.summary.image(name='dis_true', data=image, step=epoch)
        if valid_dataset is not None:
            logging.info('Time for epoch [{}/{}] is {:.2f} sec. [Training, mse: {:.6f}, dis-ssim {:.6f} ][Valid, mse: {:.6f}, dis-ssim: {:.6f}]'.format(
                epoch + 1, epochs, time.time()-start,
                generator_log_mse_high.result(), generator_log_ssim_high.result(),
                valid_gen_log_h_mse.result(), valid_gen_log_h_ssim.result()))
        else:
            logging.info('Time for epoch [{}/{}] is {:.2f} sec. [Training, mse: {:.6f}, dis-ssim {:.6f} ]'.format(
                epoch + 1, epochs, time.time()-start,
                generator_log_mse_high.result(), generator_log_ssim_high.result()))


def plot_matrix(m):
    import numpy as np
    import matplotlib.pyplot as plt
    figure = plt.figure(figsize=(10, 10))
    if len(m.shape) == 3:
        for i in range(min(9, m.shape[0])):
            ax = figure.add_subplot(3, 3, i+1)
            ax.matshow(np.squeeze(m[i, :, :]), cmap='RdBu_r')
        plt.tight_layout()
    else:
        plt.matshow(m, cmap='RdBu_r')
        plt.colorbar()
        plt.tight_layout()
    return figure


def plot_prob_matrix(m):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    figure = plt.figure(figsize=(10, 10))

    m = 1 / (1 + np.exp(-m))
    if len(m.shape) == 3:
        for i in range(min(9, m.shape[0])):
            ax = figure.add_subplot(3, 3, i+1)
            im = ax.matshow(np.squeeze(m[i, :, :]), cmap='RdBu_r')
            txt = "mean prob is {:5.4f}".format(np.mean(m[i, :, :]))
            ax.set_title(txt)
            im.set_clim(0.001, 1.001)
        plt.tight_layout()
    else:
        ax = figure.subplots()
        im = ax.matshow(m, cmap='RdBu_r', clim=[0.0, 1.0])
        norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='RdBu_r'))
        for (i, j), z in np.ndenumerate(m):
            ax.text(j, i, '{:2.2f}'.format(z), ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    return figure


def plot_to_image(figure):
    import io
    import matplotlib.pyplot as plt
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

# data from ftp://cooler.csail.mit.edu/coolers/hg19/
def train(train_data, valid_data, len_size, scale, EPOCHS, root_path='./', load_model_dir=None, saved_model_dir=None, log_dir=None, summary=False):
    if log_dir is None:
        log_dir = os.path.join(root_path, 'logs', 'model')
        os.makedirs(log_dir, exist_ok=True)
    logging.info(train_data)
    logging.info(valid_data)
    # get generator model and discriminator model
    Gen = model.make_generator_model(len_high_size=len_size, scale=scale)
    Dis = model.make_discriminator_model(len_high_size=len_size, scale=scale)
    if load_model_dir is not None:
    #load_model_dir = os.path.join(root_path, 'EnHiC', 'saved_model')
        file_path = os.path.join(load_model_dir, 'gen_model_'+str(len_size), 'gen_weights')
        if os.path.exists(file_path):
            Gen.load_weights(file_path)
        else:
            logging.info("generator doesn't exist. create a new one.")
        file_path = os.path.join(load_model_dir, 'dis_model_'+str(len_size), 'dis_weights')
        if os.path.exists(file_path):
            Dis.load_weights(file_path)
        else:
            logging.info("discriminator model doesn't exist. create a new one")

    if summary:
        logging.info(Gen.summary())
        tf.keras.utils.plot_model(Gen, to_file='G.png', show_shapes=True)
        logging.info(Dis.summary())
        tf.keras.utils.plot_model(Dis, to_file='D.png', show_shapes=True)

    if saved_model_dir is None:
        saved_model_dir = os.path.join(root_path, 'saved_model')
    os.makedirs(saved_model_dir, exist_ok=True)

    run_fit(Gen, Dis, train_data, EPOCHS, len_size, scale, valid_data, log_dir=log_dir, saved_model_dir=saved_model_dir)

    file_path = os.path.join(
        saved_model_dir, 'gen_model_'+str(len_size), 'gen_weights')
    Gen.save_weights(file_path)

    file_path = os.path.join(
        saved_model_dir, 'dis_model_'+str(len_size), 'dis_weights')
    Dis.save_weights(file_path)


def predict(model_path, len_size, scale, ds):
    # get generator model
    if model_path is None:
        gan_model_weights_path = './saved_model/gen_model_' + \
            str(len_size)+'/gen_weights'
    else:
        gan_model_weights_path = model_path
    Generator = model.make_generator_model(len_high_size=len_size, scale=scale)
    Generator.load_weights(gan_model_weights_path)
    print('Generator: \n', Generator)

    prediction = None
    for i, input_data in enumerate(ds):
        [_, _, tmp, _, _] = Generator(input_data, training=False)
        if prediction is None:
            prediction = tmp.numpy()
        else:
            prediction = np.concatenate( (prediction, tmp.numpy()), axis=0)

    print('prediction shape: {}'.format(prediction.shape))
    return  prediction