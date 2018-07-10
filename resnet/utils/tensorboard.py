import win32clipboard
import subprocess
import psutil
import os


class TensorBoard(object):

    def make(paths, names=None, host='127.0.0.1', _print=True, copy=True):
        assert isinstance(paths, list), 'Paths not type list'
        assert isinstance(names, list) or isinstance(names, type(None)), 'Names not type list or None'
        command = 'tensorboard'
        command += ' '

        if not names:
            names = [None for _ in range(len(paths))]

        is_first = True
        for name, path in zip(names, paths):
            if is_first:
                command += '--logdir='
                is_first = False
            else:
                command += ','
            if name:
                command += name + ':'
            command += path

        if host:
            command += ' '
            command += '--host' + ' ' + host

        if _print:
            print('> {}'.format(command))
        if copy:
            TensorBoard.copy(command)
        return command

    def copy(command):
        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardText(command)
        win32clipboard.CloseClipboard()


if __name__ == '__main__':
    paths = [r'C:\Users\parth\Documents\GitHub\Facial-Recognition\alexnet\tmp\tensorboard\third_training',
             r'C:\Users\parth\Documents\GitHub\Facial-Recognition\resnet\v2\tmp\tensorboard\001']
    names = ['AlexNet', 'ResNet']
    host = '127.0.0.1'

    tb = TensorBoard.start(paths=paths,
                           names=names,
                           host=host)
