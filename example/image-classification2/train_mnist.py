import find_mxnet
import mxnet as mx
import argparse
import os, sys
import train_model
import threading
from os.path import dirname, join

def _download(data_dir):
    if not os.path.isdir(data_dir):
        os.system("mkdir " + data_dir)
    os.chdir(data_dir)
    if (not os.path.exists('train-images-idx3-ubyte')) or \
       (not os.path.exists('train-labels-idx1-ubyte')) or \
       (not os.path.exists('t10k-images-idx3-ubyte')) or \
       (not os.path.exists('t10k-labels-idx1-ubyte')):
        os.system("wget http://data.dmlc.ml/mxnet/data/mnist.zip")
        os.system("unzip -u mnist.zip; rm mnist.zip")
    os.chdir("..")

def get_loc(data, attr={'lr_mult':'0.01'}):
    """
    the localisation network in lenet-stn, it will increase acc about more than 1%,
    when num-epoch >=15
    """
    loc = mx.symbol.Convolution(data=data, num_filter=30, kernel=(5, 5), stride=(2,2))
    loc = mx.symbol.Activation(data = loc, act_type='relu')
    loc = mx.symbol.Pooling(data=loc, kernel=(2, 2), stride=(2, 2), pool_type='max')
    loc = mx.symbol.Convolution(data=loc, num_filter=60, kernel=(3, 3), stride=(1,1), pad=(1, 1))
    loc = mx.symbol.Activation(data = loc, act_type='relu')
    loc = mx.symbol.Pooling(data=loc, global_pool=True, kernel=(2, 2), pool_type='avg')
    loc = mx.symbol.Flatten(data=loc)
    loc = mx.symbol.FullyConnected(data=loc, num_hidden=6, name="stn_loc", attr=attr)
    return loc

def get_mlp():
    """
    multi-layer perceptron
    """
    data = mx.symbol.Variable('data')
    fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
    fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
    act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
    fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)
    mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')
    return mlp

def get_lenet(add_stn=False):
    """
    LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick
    Haffner. "Gradient-based learning applied to document recognition."
    Proceedings of the IEEE (1998)
    """
    data = mx.symbol.Variable('data')
    if(add_stn):
        data = mx.sym.SpatialTransformer(data=data, loc=get_loc(data), target_shape = (28,28),
                                         transform_type="affine", sampler_type="bilinear")
    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
    tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
    pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
    tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
    pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # first fullc
    flatten = mx.symbol.Flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
    # second fullc
    fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
    # loss
    lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    return lenet

def get_iterator(data_shape):
    def get_iterator_impl(args):
        data_dir = args.data_dir
        if '://' not in args.data_dir:
            _download(args.data_dir)
        flat = False if len(data_shape) == 3 else True

        train           = mx.io.MNISTIter(
            image       = data_dir + "train-images-idx3-ubyte",
            label       = data_dir + "train-labels-idx1-ubyte",
            input_shape = data_shape,
            batch_size  = args.batch_size,
            shuffle     = True,
            flat        = flat,
            num_parts   = args.num_table_threads,
            part_index  = args.client_id)

        val = mx.io.MNISTIter(
            image       = data_dir + "t10k-images-idx3-ubyte",
            label       = data_dir + "t10k-labels-idx1-ubyte",
            input_shape = data_shape,
            batch_size  = args.batch_size,
            flat        = flat,
            num_parts   = args.num_table_threads,
            part_index  = args.client_id)

        return (train, val)
    return get_iterator_impl

def parse_args():
    parser = argparse.ArgumentParser(description='train an image classifer on mnist')
    parser.add_argument('--network', type=str, default='mlp',
                        choices = ['mlp', 'lenet', 'lenet-stn'],
                        help = 'the cnn to use')
    parser.add_argument('--data-dir', type=str, default='mnist/',
                        help='the input data directory')
    parser.add_argument('--gpus', type=str,
                        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--num-examples', type=int, default=60000,
                        help='the number of training examples')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='the batch size')
    parser.add_argument('--lr', type=float, default=.1,
                        help='the initial learning rate')
    parser.add_argument('--model-prefix', type=str,
                        help='the prefix of the model to load/save')
    parser.add_argument('--save-model-prefix', type=str,
                        help='the prefix of the model to save')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='the number of training epochs')
    parser.add_argument('--load-epoch', type=int,
                        help="load the model on an epoch using the model-prefix")
    parser.add_argument('--lr-factor', type=float, default=1,
                        help='times the lr with a factor for every lr-factor-epoch epoch')
    parser.add_argument('--lr-factor-epoch', type=float, default=1,
                        help='the number of epoch to factor the lr, could be .5')
    parser.add_argument('--hostfile', type=str, default='hosts',
                        help='hostfile')
    parser.add_argument('--num_table_threads', type=int,
                        help='num_table_threads')
    parser.add_argument('--num_clients', type=int,
                        help='num_clients')
    parser.add_argument('--svb', type=bool, default=False,
                        help='svb')
    parser.add_argument('--client_id', type=int,
                        help='client_id')
    return parser.parse_args()

def train(args, thread_id):
    if args.network == 'mlp':
        data_shape = (784, )
        net = get_mlp()
    elif args.network == 'lenet-stn':
        data_shape = (1, 28, 28)
        net = get_lenet(True)
    else:
        data_shape = (1, 28, 28)
        net = get_lenet()

    # train
    train_model.fit(args, net, get_iterator(data_shape), thread_id)

if __name__ == '__main__':
    args = parse_args()

    # Initialize PS
    print("Initializing PS environment\n")
    mx.pstable.register_dense_row(0, 0)
    cur_dir = dirname(os.path.realpath(__file__))
    hostfile = join(cur_dir, args.hostfile)
    # Get host IPs
    with open(hostfile, "r") as f:
        hostlines = f.read().splitlines()
    ids = [int(line.split()[0]) for line in hostlines]
    hosts = [line.split()[1] for line in hostlines]
    ports = [line.split()[2] for line in hostlines]
    mx.pstable.init(num_local_app_threads=args.num_table_threads+1,
                 num_comm_channels_per_client=args.num_clients,
                 num_total_clients=args.num_clients, num_hosts=len(ids),
                 num_tables=1, client_id=args.client_id,
                 ids=ids, hosts=hosts, ports=ports)

    mx.pstable.create_table(0, oplog_capacity=100, table_staleness=0, row_capacity=1024*1024,
                         row_oplog_type=0, oplog_dense_serialized=False,
                         dense_row_oplog_capacity=100, process_cache_capacity=100)

    print("Tables get ready\n")
    mx.pstable.create_table_done()
    print("PS initialization done.\n")
    if args.num_clients > 1 and args.svb:
        use_svb = True

    # Train
    print("Starting with %d worker threads on client %d\n" %
          (args.num_table_threads, args.client_id))
    def train_func(args, thread_id):
        """Thread entry"""
        while True:
            mx.pstable.register_thread()
            train(args, thread_id)
            mx.pstable.deregister_thread()

    train_threads = [threading.Thread(target=train_func, args=[args, i]) \
                    for i in range(args.num_table_threads)]
    for thread in train_threads:
        thread.setDaemon(True)
        thread.start()

    mx.pstable.wait_thread_register()

    # SVB
    def svb_func(args1, thread_id):
        """Thread entry"""
        while True:
            print args1,thread_id

    if use_svb:
        svb_worker_thread = threading.Thread(target=svb_func, args=[args, i])
        svb_worker_thread.setDaemon(True)
        svb_worker_thread.start()

    # Finish
    for thread in train_threads:
        thread.join()
    if use_svb:
        svb_worker_thread.join()
    print("Optimization Done.\n")
    mx.pstable.shut_down()
    print("NN finished and shut down!\n")
