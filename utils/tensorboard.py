import win32clipboard
import os


class Tensorboard(object):

    def make(paths, names=None, host='127.0.0.1', _print=True):
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

        return command


if __name__ == '__main__':

    tensorboard_paths = [r'C:\Users\parth\Documents\GitHub\Facial-Recognition\tmp\tensorboard\013',
                         r'C:\Users\parth\Documents\GitHub\Facial-Recognition\tmp\tensorboard\014']
    tensorboard_names = ['full-images', 'augment-images']
    TensorBoard.make(paths=tensorboard_paths,
                     names=tensorboard_names,
                     host='127.0.0.1',
                     _print=True)
