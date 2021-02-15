import argparse
import glob
import os
import pathlib
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str, help='RSAtrace3D directory', default='RSAtrace3D')
    parser.add_argument('-p', '--pull', action='store_true')
    parser.add_argument('-r', '--remove', action='store_true')
    args = parser.parse_args()

    assert os.path.isdir(args.directory), 'Indicate RSAtrace3D directory: use "--directory" option.'

    if args.pull == True:
        if args.remove == False:
            print(f'the mod files in RSAtrace3D directory will be restored in this directory.')
        else:
            print(f'the mod files in RSAtrace3D directory will be restored in this directory and will be removed.')
    elif args.remove == True:
        print(f'the mod files in RSAtrace3D directory will be removed.')
    else:
        print(f'the mod files in this directory will be copied in RSAtrace3D directory.')

    files = glob.glob(os.path.join(os.path.dirname(__file__), 'mod/**'), recursive=True)

    for f in files:
        rel_path = pathlib.Path(f).relative_to(os.path.dirname(__file__))
        out_path = os.path.join(args.directory, rel_path)
        
        if args.pull == False:
            if args.remove == False:
                if os.path.isdir(f):
                    os.makedirs(out_path, exist_ok=True)
                    continue
                shutil.copyfile(f, out_path)
                continue
        else:
            if os.path.isdir(f):
                continue
            shutil.copyfile(out_path, f)

        if args.remove:
            if os.path.isdir(out_path) or not os.path.isfile(out_path):
                continue
            os.remove(out_path)
            try:
                os.rmdir(os.path.dirname(out_path))
            except:
                pass
            