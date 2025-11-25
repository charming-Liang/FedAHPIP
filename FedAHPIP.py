"""
FedAHPIP: Federated Learning with Selective Homomorphic Encryption.
"""
import FedAHPIP_trainer
import FedAHPIP_client
import FedAHPIP_server

from FedAHPIP_callbacks import FedAHPIPCallback


def main():
    """A Plato federated learning training session using selective homomorphic encryption."""
    trainer = FedAHPIP_trainer.Trainer
    client = FedAHPIP_client.Client(trainer=trainer, callbacks=[FedAHPIPCallback])
    server = FedAHPIP_server.Server(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    
    main()
