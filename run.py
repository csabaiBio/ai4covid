from src.data import generate_data
import wandb
import hydra


@hydra.main(config_path='src/conf', config_name='train')
def run(config):
    #wandb.init(project=config.project, entity='elte-ai4covid')

    datasets = generate_data(config)

    train_dataset = datasets["train_dataset"]
    for sample in train_dataset:
        continue

if __name__ == "__main__":
    run()