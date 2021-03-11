import argparse


def set_gpu_ids(config):
    str_ids = config.gpu_ids.split(',')
    config.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            config.gpu_ids.append(id)
    if len(config.gpu_ids) > 0:
        torch.cuda.set_device(config.gpu_ids[0])
    return config


def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--display_freq', type=int, default=400, 
                        help='freq of uploading to visdom')
    
    parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
    parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
    parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
    parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
    
    # TODO: REMOVE
    parser.add_argument('--update_html_freq', type=int, default=1000, 
                        help='frequency of saving training results to html')
    
    parser.add_argument('--print_freq', type=int, default=100, 
                        help='freq of printing results')

    # network saving and loading parameters
    parser.add_argument('--save_latest_freq', type=int, default=5000, 
                        help='freq of saving')

    # TODO: Try to simplify save
    parser.add_argument('--save_by_iter', action='store_true', 
                        help='save model by iter?')
    parser.add_argument('--continue_train', action='store_true', 
                        help='continue from epoch')
    parser.add_argument('--epoch_count', type=int, default=1, 
                        help='starting epoch count')

    # training parameters
    parser.add_argument('--n_epochs', type=int, default=100, 
                        help='number epochs with initial lr')
    parser.add_argument('--n_epochs_decay', type=int, default=100, 
                        help='number of epochs to decay lr')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='adam momentum')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='starting rl')

    parser.add_argument('--pool_size', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1) 
    parser.add_argument('--lambda_A', type=float, default=10.0)
    parser.add_argument('--lambda_B', type=float, default=10.0)    
    parser.add_argument('--lambda_identity', type=float, default=0.5)    

    parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids; use -1 for CPU')

    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--dataroot', default='./datasets/stylemonet', 
                        help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--name', type=str, default='stylemonet', 
                        help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
    parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
    

    # TEST
    parser.add_argument('--model_suffix', type=str, default='', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')
    parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
    parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
    parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
    
    config = parser.parse_args()
    config = set_gpu_ids(config)
    return config