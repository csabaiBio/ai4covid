import wandb
import hydra
from src.image_proc import ImageProcessor
from src.table_proc import TableProcessor
import pandas as pd
import os
from tqdm import tqdm


@hydra.main(config_path='src/conf', config_name='base')
def run(config):
    #wandb.init(project=config.project, entity='elte-ai4covid')

    img_processor = ImageProcessor(config=config)
    table_prcessor = TableProcessor(config=config)

    df = pd.read_excel(config.test_table_path)
    prognosis = df.pop('Prognosis').to_numpy().flatten()
    df.drop(['Death'], axis=1, inplace=True)

    for idx in tqdm(range(len(df))):
        row=df.iloc[[idx]]
        image_file, processed_row = table_prcessor.line_impute_population_average(row)
        img_path = os.path.join(config.image_base_path, image_file)
        _ = img_processor.process_image(img_path)


if __name__ == '__main__':
    run()