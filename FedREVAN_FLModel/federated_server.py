import flwr as fl
import sys
import numpy as np

class SaveModelStrategy(fl.server.strategy.FedAvg):

    def aggregate_fit(
        self,
        rnd,
        results,
        failures,

    ):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights

# Create strategy and run server
strategy = SaveModelStrategy()

# Start Flower server for three rounds of federated learning
fl.server.start_server(
        server_address = '0.0.0.0:8080',
        config=fl.server.ServerConfig(num_rounds=5),
        grpc_max_message_length = 1024*1024*1024,
        strategy = strategy,
)

# strategy = fl.server.strategy.FedAvg()
#
# fl.server.start_server(
#     server_address = '0.0.0.0:44466',
#     config=fl.server.ServerConfig(num_rounds=3),
#     strategy=strategy,
#
# )