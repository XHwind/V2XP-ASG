import numpy as np
import torch
import copy

import opencood.tools.train_utils as train_utils
import opencood.hypes_yaml.yaml_utils as yaml_utils
import opencood.tools.infrence_utils as inference_utils
import opencood.utils.eval_utils as eval_utils


class ModelWrapper(object):
    """
    This class is a wrapper class to initiate opencood model, load params,
    detect, and evaluate. In such way, SG can easily use the trained models.

    Parameters
    ----------
    opt : argparser
        Input argparser for saved model path and fusion method.

    Attributes
    ----------
    model : torch.nn.Module
        The loaded pytorch model for cooperative 3D Detection.

    hypes : dict
        The hype parameters of the model.

    fusion_method : str
        Indicate which fusion method to use.
    """

    def __init__(self, opt):
        self.hypes = yaml_utils.load_yaml(None, opt)
        self.fusion_method = opt.fusion_method

        model = train_utils.create_model(self.hypes)

        if opt.cuda and torch.cuda.is_available():
            model.cuda()

        self.device = torch.device('cuda' if torch.cuda.is_available()
                                             and opt.cuda else 'cpu')

        print('Loading Model from checkpoint')
        _, self.model = train_utils.load_saved_model(opt.model_dir,
                                                     model)
        self.model.eval()

    def inference(self, batch_data, dataset):
        """
        Given the observation, run inference using the trained model.

        Parameters
        ----------
        batch_data : dict
            The formatted input data (torch.Tensor format).

        dataset : opencood.Dataset
            Early/Late/Intermedaite dataset class.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        pred_score : torch.Tensor
            The tensor prediction score for each bounding box.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, self.device)

            if self.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor, _ = \
                    inference_utils.inference_late_fusion(batch_data,
                                                          self.model,
                                                          dataset)
            elif self.fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor, _ = \
                    inference_utils.inference_early_fusion(batch_data,
                                                           self.model,
                                                           dataset)
            elif self.fusion_method == 'intermediate':
                pred_box_tensor, pred_score, gt_box_tensor, _ = \
                    inference_utils.inference_intermediate_fusion(batch_data,
                                                                  self.model,
                                                                  dataset)
            elif self.fusion_method == 'nofusion':
                pred_box_tensor, pred_score, gt_box_tensor, _ = \
                    inference_utils.inference_no_fusion(batch_data,
                                                        self.model,
                                                        dataset)

            else:
                raise NotImplementedError('Only early, late and intermediate'
                                          'fusion is supported.')

        return pred_box_tensor, pred_score, gt_box_tensor

    def get_attention_weight(self, batch_data, dataset):
        """
        Given the observation, run inference using the trained model.

        Parameters
        ----------
        batch_data : dict
            The formatted input data (torch.Tensor format).

        dataset : opencood.Dataset
            Early/Late/Intermedaite dataset class.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        pred_score : torch.Tensor
            The tensor prediction score for each bounding box.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        attn : torch.Tensor
            The tensor of attention weights.
        """
        assert self.fusion_method == 'intermediate'
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, self.device)
            pred_box_tensor, pred_score, gt_box_tensor, output_dict = \
                inference_utils.inference_intermediate_fusion(batch_data,
                                                              self.model,
                                                              dataset)
        return pred_box_tensor, pred_score, gt_box_tensor, \
               output_dict["ego"]["attn"]

    def evaluate(self, pred_box_tensor, pred_score, gt_box_tensor):
        """
        Given the prediction and gt, compute APs.

        Parameters
        ----------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        pred_score : torch.Tensor
            The tensor prediction score for each bounding box.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.

        Returns
        -------
        ap_70 : float
            AP 70 score.
        """
        # Create the dictionary for evaluation
        iou_threshs = [0.3, 0.5, 0.7]
        result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0},
                       0.5: {'tp': [], 'fp': [], 'gt': 0},
                       0.7: {'tp': [], 'fp': [], 'gt': 0}}
        APs = {}
        for iou_thresh in iou_threshs:
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       iou_thresh)
            # The calculate_ap will modify the list of result_stat in place.
            # Thus we provide a copy of it.
            ap, _, _ = eval_utils.calculate_ap(copy.deepcopy(result_stat),
                                               iou_thresh)
            APs[iou_thresh] = ap

        return APs, result_stat

    def visualize(self, pred_box_tensor, gt_box_tensor, origin_lidar, dataset,
                  show_vis=True, vis_save_path='', color_dict=None):
        """
        Visualzie the prediction result on lidar map.

        Parameters
        ----------
        pred_box_tensor : torch.Tensor
             The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        origin_lidar : torch.Tensor
            (n, 4) lidar point cloud
        dataset : opencood.dataset
            Early/late/intermeidate dataset class.
        show_vis : bool
            Whether to show vis in open3d
        vis_save_path : str
            Visualization save path
        """
        dataset.visualize_result(pred_box_tensor,
                                 gt_box_tensor,
                                 origin_lidar,
                                 show_vis,
                                 vis_save_path,
                                 dataset=dataset)


if __name__ == '__main__':
    import argparse
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.data_utils.datasets import EarlyFusionDataset

    # give a argparser
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str,
                        default='logs/point_pillar_early',
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='early',
                        help='late, early or intermediate')
    parser.add_argument('--cuda', action='store_false',
                        help='late, early or intermediate')
    opt = parser.parse_args()

    params = load_yaml('hypes_yaml/point_pillar_early_fusion.yaml', opt)

    opencda_dataset = EarlyFusionDataset(params, train=True, visualize=True)
    observation = {
        '-1': {'lidar_np': np.random.randn(3962, 4),
               'lidar_pose': [-103.89463806152344, 58.31967544555664,
                              2.3705999851226807, 0.0, -89.35775756835938,
                              0.0],
               'ego': True}
    }
    gt_info = {
        '1': {
            'angle': [0, 10, 0],
            'center': [0, 0, 0.7],
            'extent': [2.4, 1.06, 0.75],
            'location': [-246, 250, 0.03]
        }
    }
    processed_data = opencda_dataset.data_reformat(observation, gt_info)
    tensor_data = opencda_dataset.collate_batch_test([processed_data])

    # initiate model
    model = ModelWrapper(opt)
    # perform inference
    pred_box_tensor, pred_score, gt_box_tensor = \
        model.inference(tensor_data,
                        opencda_dataset)
    # evaluate
    ap_dict, result_stats = model.evaluate(pred_box_tensor, pred_score,
                                           gt_box_tensor)
    print(ap_dict)

    # visualize
    model.visualize(pred_box_tensor,
                    gt_box_tensor,
                    tensor_data['ego']['origin_lidar'][0],
                    opencda_dataset)
