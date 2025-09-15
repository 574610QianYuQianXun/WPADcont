import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0,
                        help="GPU ID, -1 for CPU")

    parser.add_argument('--dataset', type=str, default='MNIST',
                        help="name of dataset: MNIST, FEMNIST, CIFAR10")

    parser.add_argument('--num_classes', type=int, default=10,
                        help="number of classes: 10(MNIST,CIFAR10), 62(FEMNIST)")

    parser.add_argument('--model', type=str, default='CNNMnist',
                        help='ResNet18, CNNMnist')

    parser.add_argument('--malicious', type=float, default=0.2,
                        help="Proportion of malicious clients")

    parser.add_argument('--attack_epoch', type=float, default=3,
                        help="Attack start time")

    parser.add_argument('--watermark', type=bool, default=False,
                        help="Whether to add a watermark")

    parser.add_argument('--defense', type=str, default='no',
                        help="Choose a defense method:vae, fld")

    parser.add_argument('--attack_type', type=str, default='How_backdoor',
                        help="How_backdoor")

    parser.add_argument('--agg', type=str, default='FedAvg',choices=['FedAvg', 'FoolsGold'],
                        help="FedAvg, Trimmed_mean, FoolsGold")

    parser.add_argument('--a', type=float, default=0.5,     # attack_epoch
                        help="The no-iid degree of data")

    parser.add_argument('--clients', default=100,
                        help='number of clients')

    parser.add_argument('--epochs', type=int, default=100,
                        help="rounds of training")

    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size of client")

    parser.add_argument('--bs', type=int, default=128,
                        help="test batch size")

    parser.add_argument('--lr', type=float, default=1e-3,           # 1e-3
                        help="learning rate")

    parser.add_argument('--momentum', type=float, default=0.5,
                        help="SGD momentum")

    parser.add_argument('--local_ep', type=int, default=1,
                        help="the number of local epochs: E")        # 每个客户端只训练一次

    args = parser.parse_args()

    return args


