from typing import List, Tuple
from itertools import chain, product

import numpy as np
import pandas as pd


class ApplianceOptimizer:
    def __init__(self, trained_model, features_config):
        self.trained_model = trained_model
        self.features_config = features_config
        self.controllable_parameters = features_config["optimization_features"]

    def optimize_appliance(self, feature_series):
        current_values_for_controllable_parameters = feature_series[
            self.controllable_parameters
        ]
        grid_with_only_controllable_features = self.make_grid(
            current_values=current_values_for_controllable_parameters
        )
        grid_with_all_features = add_uncontrollable_parameters(
            grid_with_only_controllable_features,
            feature_series,
            self.features_config["features_to_normalize"]
            + self.features_config["raw_features"],
        )
        predictions = self.trained_model.predict(grid_with_all_features)["yhat"]
        return min(predictions)

    def make_grid(
        self,
        current_values: List[float],
        step=0.3,
        parameter_range=1.5,
    ):
        """
        Each parameter to optimize is a temperature so we will apply the same range for each i.e 16°C-20°C
        """
        values = []
        for current_value in current_values:

            # Create ranges of values for each parameter as per grid range
            parameter_range_list = chain(
                *[
                    [
                        current_value + micro_step,
                        current_value - micro_step,
                    ]
                    for micro_step in np.arange(0, parameter_range, step)
                ]
            )

            values.append(parameter_range_list)

        # Create grid with all combinations of parameters
        return pd.DataFrame(
            product(*values),
            columns=self.controllable_parameters,
        )


def add_uncontrollable_parameters(
    grid: pd.DataFrame, features: pd.Series, uncontrollable_params: List[str]
):
    """
    Add back uncontrollable parameters
    """
    for var in uncontrollable_params:
        grid[var] = features[var]
    return grid
