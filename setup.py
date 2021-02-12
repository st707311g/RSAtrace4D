import argparse
import glob
import os
import pathlib
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str, help='RSAtrace3D directory', default='RSAtrace3D')
    args = parser.parse_args()

    assert os.path.isdir(args.directory), 'Indicate RSAtrace3D directory: use "--directory" option.'

    files = glob.glob(os.path.join(os.path.dirname(__file__), 'mod/**'), recursive=True)

    for f in files:
        rel_path = pathlib.Path(f).relative_to(os.path.dirname(__file__))
        out_path = os.path.join(args.directory, rel_path)
        if os.path.isdir(f):
            os.makedirs(out_path, exist_ok=True)
            continue
        shutil.copyfile(f, out_path)