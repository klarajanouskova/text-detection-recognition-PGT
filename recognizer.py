import torch
import os
import numpy as np
import cv2
import gc

import math

from utils_ocr import CTCLabelConverter, AttnLabelConverter, Averager
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from ocr_dataset.dataset import align_images

from model_ocr import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda'


def get_crop_from_im(im, rot_bbox):
    im_h, im_w, _ = im.shape

    width = int(
        np.round(math.sqrt((rot_bbox[2][0] - rot_bbox[1][0]) ** 2 + (rot_bbox[2][1] - rot_bbox[1][1]) ** 2)))
    height = int(
        np.round(math.sqrt((rot_bbox[3][1] - rot_bbox[2][1]) ** 2 + (rot_bbox[3][0] - rot_bbox[2][0]) ** 2)))

    # coordinates of the straight rectangle
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(rot_bbox.astype(np.float32), dst_pts.astype(np.float32))

    crop = cv2.warpPerspective(im, M, (width, height))
    return crop

class Recognizer():
    def __init__(self, opt):
        self.load_ocr_model(opt)
        self.batch_max_length = opt.batch_max_length
        self.bs = opt.batch_size
        self.H = opt.imgH
        self.W = opt.imgW
        self.in_ch = opt.input_channel
        self.toTensor = ToTensor()


    def load_ocr_model(self, opt):
        """ model configuration """
        if 'CTC' in  opt.Prediction:
            self.converter = CTCLabelConverter(opt.character)
        else:
            self.converter = AttnLabelConverter(opt.character)
        opt.num_class = len(self.converter.character)

        # opt.input_channel = 3
        model = Model(opt)
        print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
              opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
              opt.SequenceModeling, opt.Prediction)

        model = torch.nn.DataParallel(model).to(device)

        # model = model.to(device)


        # load model
        print('loading pretrained model from %s' % opt.saved_model)
        model.load_state_dict(torch.load(opt.saved_model, map_location=device))
        # print(model)


        """ evaluation """
        model.eval()

        self.model = model

    def predict(self, image_tensors, full=False):
        with torch.no_grad():
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([self.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, self.batch_max_length + 1).fill_(0).to(device)


            preds = self.model(image, text_for_pred, is_train=False)

            # not sure about the text for pred here TODO check this
            preds = preds[:, :text_for_pred.shape[1] - 1, :]

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index, length_for_pred)

            # calculate accuracy & confidence score
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            confidence_score_list = []
            output_strings = []

            # calculate confidence score (= multiply of pred_max_prob)
            for (pred_max_prob, pred) in zip(preds_max_prob, preds_str):
                try:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS] # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                except:
                    confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
                confidence_score_list.append(confidence_score)
                output_strings.append(pred)
                # print(pred, gt, pred==gt, confidence_score)
        if full:
            return output_strings, confidence_score_list, preds_prob
        else:
            return output_strings, confidence_score_list

    def read_im(self, im, detections, full=False):
        im_h, im_w, _ = im.shape

        crops = []
        for detection in detections:
            if np.any(detection < 0) or np.any(detection[:, 0] >= im_w) or np.any(detection[:, 1] >= im_h):
                detection[:, 0] = detection[:, 0].clip(0, im_w - 1)
                detection[:, 1] = detection[:, 1].clip(0, im_h - 1)

            crop = get_crop_from_im(im, detection)
            if self.in_ch == 1:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crops.append(crop)

        n = len(crops) // self.bs

        predictions, confidences = [], []
        if full:
            predictions_full = []

        for i in range(n + 1):
            if i == n:
                if len(crops) % self.bs != 0:
                    batch_ims = crops[i * self.bs:]
                else:
                    break
            else:
                batch_ims = crops[i * self.bs:(i+1) * self.bs]
            batch = align_images(batch_ims, self.toTensor, self.H, self.W, True)
            preds, confs, preds_full = self.predict(batch, full=True)
            predictions.extend(preds)
            confidences.extend(confs)
            if full:
                predictions_full.extend(preds_full)

        if full:
            predictions_full = torch.stack(predictions_full) if len(predictions_full) > 0 else torch.tensor([])
            return predictions, confidences, predictions_full
        else:
            return predictions, confidences