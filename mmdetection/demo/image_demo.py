import os
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
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    img_list = os.listdir(args.img)
    for i, img in enumerate(img_list):
        # if i < 6:
        #     continue
        if i == 109:
            break
        if i not in [34, 36, 64, 108]:
            continue
        img = os.path.join(args.img, img)
        result = inference_detector(model, img)
        # show the results
        show_result_pyplot(model, img, result, score_thr=args.score_thr, name=i+1)

if __name__ == '__main__':
    main()
