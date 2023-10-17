from asg.core.adversarial_optimization.bayesian_optimization import \
    BayesianOptimization
from asg.core.adversarial_optimization.random_search import RandomSearch
from asg.core.adversarial_optimization.genetic_algorithm import \
    GeneticAlgorithm

__all__ = {
    'BayesianOptimization': BayesianOptimization,
    'RandomSearch': RandomSearch,
    'GeneticAlgorithm': GeneticAlgorithm
}


def build_adversarial_model(opt, adversarial_config):
    adversarial_model_name = adversarial_config["core_method"]

    error_message = f"{adversarial_model_name} is not found." \
                    f"Please set the adversarial core_method_name from the " \
                    f"list [\"RandomSearch\", \"BayesianOptimization\", \"GeneticAlgorithm\"]"
    assert adversarial_model_name in ["RandomSearch", "BayesianOptimization",
                                      "GeneticAlgorithm"], error_message

    adversarial_model = __all__[adversarial_model_name](opt,
                                                        adversarial_config)
    return adversarial_model
