from os.path import join, abspath
from os import getcwd, pardir

PARENT_DIRECTORY = abspath(join(getcwd(), pardir))
DATA_DIR = join(PARENT_DIRECTORY, 'data')