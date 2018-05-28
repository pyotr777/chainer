from __future__ import print_function
import argparse, time
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import chainer

import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.training import triggers
from chainer.training import util

from chainer.datasets import get_cifar10
from chainer.datasets import get_cifar100

#import models
import os

# import cupy as cp
# import random
# import numpy as np

base_time=0

def main():
    global base_time
    parser = argparse.ArgumentParser(description='Chainer CIFAR example:')
    parser.add_argument('--dataset', '-d', default='cifar10',
                        help='The dataset to use: cifar10 or cifar100')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--learnrate', '-l', type=float, default=0.05,
                        help='Learning rate for SGD')
    parser.add_argument('--epoch', '-e', type=int, default=10000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--early-stopping', type=str,
                        help='Metric to watch for early stopping')
    parser.add_argument('--host', type=str, help='Host name (used in log file name)')
    parser.add_argument('--debug', action='store_true', help='Log timing info')
    parser.add_argument('--accuracy', type=float, default=None, help='Log timing info')
    parser.add_argument('--time_limit', type=int, default=None, help="Execution time limit in seconds")
    parser.add_argument('--samples', type=int, default=None, help="Training set size")
    parser.add_argument('--iterations', type=int, default=None, help="Limit number of training iterations")
    parser.add_argument('--model', type=str, default="VGG", help="Model name in format modu")
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# b{} l{}'.format(args.batchsize,args.learnrate))
    if args.time_limit:
        print('# time_limit: {}s'.format(args.time_limit))

    if args.iterations:
        print("# Limit number of training iterations to {}".format(args.iterations))

    if args.samples:
        if args.iterations:
            print("Training set size (--samples) and iterations limit (--iterations) cannot be used together. Using iterations limit {}".format(args.iterations))


    # chainer.global_config.cudnn_deterministic = True
    # seed=0
    # random.seed(seed)
    # np.random.seed(seed)
    # cp.random.seed(seed)
    # print("Deterministic")
    print("Non-Deterministic")


    # Set up a neural network to train.
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    if args.dataset == 'cifar10':
        print('Using CIFAR10 dataset.')
        class_labels = 10
        train, test = get_cifar10()
    elif args.dataset == 'cifar100':
        print('Using CIFAR100 dataset.')
        class_labels = 100
        train, test = get_cifar100()
    else:
        raise RuntimeError('Invalid dataset choice.')

    # Set limit for training set samples
    samples=None
    if args.iterations:
        samples = args.batchsize*args.iterations
    elif args.samples:
        samples = args.samples

    if samples is not None:
        print("# {} samples".format(samples))
        train = train[:samples]


    print('')

    model_module = __import__("models."+args.model)
    model_class = args.model

    module = getattr(model_module,model_class)
    method = getattr(module,model_class)
    print("Method:",method)
    model = L.Classifier(method(class_labels))
    #model = L.Classifier(models.VGG.VGG(class_labels))
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    optimizer = chainer.optimizers.MomentumSGD(args.learnrate)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))


    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    stop_trigger = (args.epoch, 'epoch')
    # Early stopping option
    if args.early_stopping:
        stop_trigger = triggers.EarlyStoppingTrigger(
            monitor=args.early_stopping, verbose=True,
            max_trigger=(args.epoch, 'epoch'))
    if args.time_limit:
        if stop_trigger is not None:
            stop_trigger = StopTrigger(time_limit=args.time_limit,child_trigger=stop_trigger)
        else:
            stop_trigger = StopTrigger(time_limit=args.time_limit)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, stop_trigger, out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Reduce the learning rate by half every 25 epochs.
    trainer.extend(extensions.ExponentialShift('lr', 0.5),
                   trigger=(25, 'epoch'))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    #trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot at each epoch
    #trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    # NOTE: If you use the EarlyStoppingTrigger,
    #       training_length is needed to set
    #       because trainer.stop_trigger is not normal interval trigger.
    #trainer.extend(extensions.ProgressBar(
    #    training_length=(args.epoch, 'epoch')))

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)


    base_time = time.time()
    # Run the training
    trainer.run()


class StopTrigger(object):
    global base_time

    def __init__(self,time_limit=60*60*12,child_trigger=None):
        self.time_limit = time_limit
        if child_trigger is not None:
            self._child_trigger = util.get_trigger(child_trigger)
        #print("Stop by time >=",time_limit)

    def __call__(self, trainer):
        elapsed = time.time() - base_time
        #print("Elapsed:",elapsed)
        if elapsed >= self.time_limit:
                print("Time limit of {}s reached".format(self.time_limit))
                return True
        if self._child_trigger is not None:
            if self._child_trigger(trainer):
                print("Stop trigger activated: {}>={}".format(self._child_trigger.unit,self._child_trigger.period))
                return True
        return False

if __name__ == '__main__':
    main()
