import string
import argparse
import time

import torch

from recognizer import Recognizer
from ocr_dataset import RawDataset, AlignCollateNP

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(recognizer, opt):
    dataset = RawDataset(opt.im_path, opt)
    AlignCollate_evaluation = AlignCollateNP(imgH=opt.imgH, imgW=opt.imgW, labels=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_evaluation, pin_memory=True)


    # Running the model

    for i, (image_tensors, labels) in enumerate(loader):
        image = image_tensors.to(device)
        predictions = recognizer.predict(image)
        print(predictions)

if __name__ == '__main__':
    characters = string.ascii_lowercase + string.digits + string.punctuation + ' '

    parser = argparse.ArgumentParser()

    parser.add_argument('--im_path', type=str, default='demo_images',
                        help='path to the folder with images')
    parser.add_argument('--batch_ratio', type=str, default='1', help='assign ratio for each selected data in the batch')
    parser.add_argument('--eval_data', default='/home/klarka/mnt-cmp/datasets/evaluation',
                        help='path to evaluation dataset')
    parser.add_argument('--train_data', default='/home/klarka/mnt-cmp/datasets/str-training',
                        help='path to training dataset')
    parser.add_argument('--valid_data', default='/home/klarka/mnt-cmp/datasets/validation-str-b',
                        help='path to validation dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--batch_size', type=int, default=150, help='input batch size')
    parser.add_argument('--saved_model', default='../../saved_models/uber_ft1/best_accuracy.pth',
                        help="path to saved_model to evaluation")
    # ../Snake/saved_models/lmdb_datasets/best_accuracy.pth../Snake/saved_models/lmdb_datasets/best_accuracy.pth../Snake/saved_models/lmdb_datasets/best_accuracy.pth
    # parser.add_argument('--saved_model', default='/home/klarka/mnt-cmp/Snake/saved_models/add_padding/best_accuracy.pth', help="path to saved_model to evaluation")

    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')

    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=150, help='the width of the input image')
    parser.add_argument('--rgb', default='true', help='use rgb input')
    parser.add_argument('--character', type=str, default=characters, help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', default=True, help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')

    """ Model Architecture """
    parser.add_argument('--Transformation', default='TPS', type=str, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', default='ResNet', type=str,
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', default='BiLSTM', type=str, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', default='Attn', type=str, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    args = parser.parse_args()
    recognizer = Recognizer(args)

    main(recognizer, args)