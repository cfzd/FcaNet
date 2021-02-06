import os
import pickle
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    
    results = {}
    for img_dir in os.listdir(args.img):
        if not img_dir.endswith('images'):
            continue
        results[img_dir] = {}
        try:
            os.mkdir(osp.join('/home/pengyi/results', img_dir))
        except Exception:
            pass
        _img_dir = osp.join(args.img, img_dir)
        for img in os.listdir(_img_dir):
            results[img_dir][img] = {}
            _img = osp.join(args.img, img_dir, img)
            # test a single image
            result = inference_detector(model, _img)
            bboxes = result[0]
            proposal = []
            if len(bboxes[0]) == 0:
                continue
            for j in range(len(bboxes[0])):
                proposal.append(bboxes[0][j])
            proposal = np.array(proposal)
            scores = proposal[:, -1]
            inds = scores > args.score_thr
            bboxes = proposal[inds, :]
            _img = plt.imread(_img)
            for idx, bbox in enumerate(bboxes):
                bbox_int = bbox.astype(np.int32)
                left_top = (bbox_int[0], bbox_int[1])
                right_bottom = (bbox_int[2], bbox_int[3])
                plt.imsave(osp.join('/home/pengyi/results', img_dir, img.split('.')[0]+'{}.jpg'.format(idx+1)), _img[int(left_top[1]):int(right_bottom[1]), int(left_top[0]):int(right_bottom[0]), :])
                results[img_dir][img][idx+1] = bbox_int[:4]
    with open('/home/pengyi/results/mask_pos.pkl', 'wb') as f:
        pickle.dump(results, f)

            # show the results
            # show_result_pyplot(model, args.img, result, score_thr=args.score_thr)


if __name__ == '__main__':
    main()
