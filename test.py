"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from data import create_dataset
from util.visualizer import save_images
from util import html
from models.cycle_gan_test_model import CycleGANTest
import config_parser



if __name__ == '__main__':
    config = config_parser.parse()
    # hard-code some parameters for test
    config.num_threads = 0   # test code only supports num_threads = 0
    config.batch_size = 1    # test code only supports batch_size = 1
    config.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    config.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    config.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    config.dataset_mode = 'single'
    config.phase = 'test'
    config.isTrain = False
    
    dataset = create_dataset(config)  # create a dataset given opt.dataset_mode and other options
    model = CycleGANTest(config)      # create a model given opt.model and other options
    model.setup()               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(config.results_dir, config.name, '{}'.format(config.epoch))  # define the website directory
    if config.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, config.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Epoch = %s' % (config.name, config.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.

    for i, data in enumerate(dataset):
        if i >= config.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        # TODO Check removing aspect ratio is fine
        save_images(webpage, visuals, img_path, width=config.display_winsize, fake_only=True)
    webpage.save()  # save the HTML
