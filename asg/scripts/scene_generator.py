import time
import warnings

from asg.hypes_yaml.yaml_utils import load_yaml
from asg.data_utils.scene_datasets.scene_dataset import SceneDataset
from asg.core.adversarial_optimization import build_adversarial_model
from asg.utils.adversarial_utils import resume_attack, update_attack_meta, \
    setup_attack

warnings.filterwarnings("ignore")


def generate_adversarial_scenes(opt):
    scene_params = load_yaml(opt.sg_hypes_yaml)
    scene_params["opencda_format"] = opt.load_opencda_format_data
    if opt.client_port > 0:
        scene_params["world"]["client_port"] = opt.client_port

    dataset = SceneDataset(scene_params)
    attack_model = build_adversarial_model(opt, scene_params["adversarial"])

    if opt.resume_attack_path:
        saved_path = opt.resume_attack_path
        attack_meta_data, epoch, score_first_agg, score_cav_selection_agg, score_best_agg = resume_attack(
            opt.resume_attack_path)
    else:
        # If attack from scratch, we need to create a folder to save the attack
        # meta data
        saved_path, attack_meta_data, epoch, score_first_agg, score_cav_selection_agg, score_best_agg = setup_attack(
            scene_params, opt)

    epoches = min(opt.epoches, len(dataset))
    for i in range(epoch + 1, epoches):
        scene = dataset[i]
        print(f"Iteration {i}/{len(dataset)}")
        t1 = time.time()
        out = attack_model.forward(scene, i, vis_flag=opt.show_vis)

        score_first, score_cav_selection, score_best = out["score_first"], \
                                                       out[
                                                           "score_cav_selection"], \
                                                       out["score_best"]
        score_first_agg.add(score_first)
        score_cav_selection_agg.add(score_cav_selection)
        score_best_agg.add(score_best)

        attack_model.dump_data(saved_path, scene, out)
        dataset.clean()

        msg = f"{'score_first:':<20} {score_first} / {score_first_agg.get_avg_adv_cost()} \n" \
              f"{'score_cav_selection:':<20} {score_cav_selection} / {score_cav_selection_agg.get_avg_adv_cost()} \n" \
              f"{'score_best:':<20} {score_best} / {score_best_agg.get_avg_adv_cost()} \n" \
              f"Time per scene: {time.time() - t1}"
        print(msg)

        update_attack_meta(saved_path,
                           attack_meta_data,
                           i,
                           score_first_agg,
                           score_cav_selection_agg,
                           score_best_agg,
                           msg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str,
                        default='../../opencood/logs/point_pillar_early_sparse',
                        help='Continued training path')
    parser.add_argument('--sg_hypes_yaml', type=str,
                        default='../hypes_yaml/sg_random_search.yaml',
                        help='SG hyperparameter file path')
    parser.add_argument('--fusion_method', type=str,
                        default='early',
                        help='nofusion, late, early or intermediate')
    parser.add_argument('--resume_attack_path', type=str,
                        default='',
                        help='Resume attack file path')
    parser.add_argument('--cuda', action='store_false',
                        help='If set, then not use cuda')
    parser.add_argument('--load_opencda_format_data', action='store_true',
                        help='Set if loading OpenCDA format data')
    parser.add_argument('--use_original_cavs', action='store_true',
                        help='Set if using the original CAVs combination'
                             'as the initial combination')
    parser.add_argument('--discard_cav_selection', action='store_true',
                        help='Set if not changing the CAV combination during'
                             'optimization')
    parser.add_argument('--v2x', type=str,
                        default='v2i',
                        help="Choose v2x type from ['v2v', 'v2i', 'v2x']")
    parser.add_argument('--agent_weight_dir', type=str,
                        default='',
                        help='Folder for saved agent list')
    parser.add_argument('--epoches', type=int,
                        default=10000,
                        help="scene number for running")
    parser.add_argument('--client_port', type=int,
                        default=-1,
                        help='CARLA client port number. '
                             'If value < 0 (default -1), use the port in yaml file.'
                             'If value > 0, use the input port')
    parser.add_argument('--show_vis', action='store_true',
                        help='Set to enable visualization')

    opt = parser.parse_args()

    generate_adversarial_scenes(opt)
