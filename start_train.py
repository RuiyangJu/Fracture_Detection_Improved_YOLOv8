import argparse
from ultralytics import YOLO

def parse_args():
    parse = argparse.ArgumentParser(description='Data Postprocess')
    parse.add_argument('--model', type=str, default=None, help='load the model')
    parse.add_argument('--data_dir', type=str, default=None, help='the dir to data')
    args = parse.parse_args()
    return args

def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(data=args.data_dir)

if __name__ == '__main__':
    main()
