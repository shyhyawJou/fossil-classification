import argparse

from utils import Split



def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', help='the root of fossil images')
    parser.add_argument('-save_dir', help='the folder where the splitted data will be saved')
    parser.add_argument('-seed', type=int, help='random seed for splitting data')
    arg = parser.parse_args()
    return arg 



if __name__ == '__main__':
    arg = get_arg()

    Split(arg.data, arg.save_dir, arg.seed)
