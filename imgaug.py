import os
import cv2
import argparse

def parse_args():
    parse = argparse.ArgumentParser(description='Data Augmentation')
    parse.add_argument('--input_img', type=str, default=None, help='Path to input images.')
    parse.add_argument('--output_img', type=str, default=None, help='Path to output images.')
    parse.add_argument('--input_label', type=str, default=None, help='Path to input labels.')
    parse.add_argument('--output_label', type=str, default=None, help='Path to output labels.')
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    input_imgdir = args.input_img
    output_imgdir = args.output_img
    input_labeldir = args.input_label
    output_labeldir = args.output_label

    if not os.path.isdir(output_imgdir):
        os.mkdir(output_imgdir)
    
    if not os.path.isdir(output_labeldir):
        os.mkdir(output_labeldir)
        
    for root, dirs, files in os.walk(input_imgdir):
        for name in files:
            img_name = os.path.join(name)
            print(img_name)
            pic = cv2.imread(input_imgdir + img_name)
            contrast = 1.2
            brightness = 30
            pic_turn = cv2.addWeighted(pic, contrast, pic, 0, brightness)
            cv2.imwrite(output_imgdir  + img_name, pic)
            cv2.imwrite(output_imgdir + 'aug_' + img_name, pic_turn)
            cv2.waitKey(0)

    for root, dirs, files in os.walk(input_labeldir):
        for name in files:
            label_name = os.path.join(name)
            print(label_name)
            with open(input_labeldir + label_name, 'r', encoding = 'utf-8') as f:
                lines = f.readlines()
                label_transfer_name = output_labeldir + 'aug_' + label_name
                with open(label_transfer_name, 'w', encoding = 'utf-8') as transfer_f:
                    transfer_f.writelines(lines)
                with open(output_labeldir + label_name, 'w' , encoding = 'utf-8') as original_f:
                    original_f.writelines(lines)
                
