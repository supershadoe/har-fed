import logging

import torch
from flwr.client import ClientApp
from flwr.server import ServerApp
from flwr.simulation import run_simulation

from train.fl_client import client_fn
from train.fl_server import server_fn


logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(asctime)s: %(message)s',
)
logger = logging.getLogger(__name__)
logging.getLogger("flwr").setLevel(logging.WARNING)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == '__main__':
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}
    if DEVICE == "cuda":
        backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}
    client = ClientApp(client_fn=client_fn)
    server = ServerApp(server_fn=server_fn)
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=8,
        backend_config=backend_config,
    )
