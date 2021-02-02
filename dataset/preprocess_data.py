
import os
import argparse
from zipfile import ZipFile

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data pre-processor')
    parser.add_argument('root', metavar='R', type=str,
                    help='Root directory of the folder containing zip files')
    args = parser.parse_args()
    
    base_dir = args.root
    for f in os.listdir(base_dir):
        if not os.path.exists("./" + f[:-4]):
            ZipFile(os.path.join(base_dir, f)).extractall("./" + f[:-4])
        else:
            print("Skipping extraction, file existed!")
