from utils.tensorboard import Tensorboard

tensorboard_paths = [r'C:\Users\parth\Documents\GitHub\Facial-Recognition\tmp\tensorboard\013',
                     r'C:\Users\parth\Documents\GitHub\Facial-Recognition\tmp\tensorboard\014']
tensorboard_names = ['full-images', 'augment-images']

Tensorboard.make(paths=tensorboard_paths,
                 names=tensorboard_names,
                 host='127.0.0.1',
                 _print=True,
                 copy=True)
