import os


class Paths (object):
    def exists(dir):
        return os.path.exists(dir)

    def make_dir(dir):
        try:
            os.makedirs(dir)
            return True
        except:
            PermissionError('Could not make dir {}'.format(path))

    def remove_dir(path):
        try:
            os.rmdir(path)
            return True
        except:
            PermissionError('Could not remove dir {}'.format(path))
