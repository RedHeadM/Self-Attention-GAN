import argparse


def str2bool(v):
    return v.lower() in ('true')


def get_parameters():

    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--model', type=str, default='sagan', choices=['sagan', 'qgan'])
    parser.add_argument('--adv_loss', type=str, default='wgan-gp', choices=['wgan-gp', 'hinge'])
    parser.add_argument('--imsize', type=int, default=64)
    parser.add_argument('--g_num', type=int, default=5)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--lambda_gp', type=float, default=10)
    parser.add_argument('--version', type=str, default='sagan_1')

    # Training setting
    parser.add_argument('--total_step', type=int, default=1000000,
                        help='how many times to update the generator')
    parser.add_argument('--d_iters', type=float, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0004)
    parser.add_argument('--lr_decay', type=float, default=0.95)  # is not used
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.9)

    # using pretrained
    parser.add_argument('--pretrained_model', type=int, default=None)

    # Misc
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--parallel', type=str2bool, default=False)
    parser.add_argument('--dataset', type=str, default='tcn', choices=['lsun', 'celeb', 'tcn'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Path
    parser.add_argument('--image_path', type=str,
                        default='/tmp/tcn_data/push_3_labled_V10_color/videos/train')
    parser.add_argument('--log_path', type=str, default='/media/markus/DATA/tcn_traind/sagan/logs')
    parser.add_argument('--model_save_path', type=str, default='/media/markus/DATA/tcn_traind/sagan/models')
    parser.add_argument('--sample_path', type=str, default='/media/markus/DATA/tcn_traind/sagan/samples')
    parser.add_argument('--attn_path', type=str, default='/media/markus/DATA/tcn_traind/sagan/attn')

    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=100)
    parser.add_argument('--model_save_step', type=float, default=100.0)

    return parser.parse_args()
