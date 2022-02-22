import os

import hydra
import pandas as pd
from tqdm import tqdm

from src.image_proc import ImageProcessor
from src.table_proc import TableProcessor


@hydra.main(config_path="src/conf", config_name="preproc")
def run(config):
    img_processor = ImageProcessor(config=config)
    table_prcessor = TableProcessor(config=config)

    df = pd.read_excel(config.table_path)
    _ = df.pop("Prognosis").to_numpy().flatten()
    df.drop(["Death"], axis=1, inplace=True)

    for idx in tqdm(range(len(df))):
        row = df.iloc[[idx]]
        image_file, _ = table_prcessor.line_impute_population_average(row)
        img_path = os.path.join(config.image_base_path, image_file)
        _ = img_processor.process_image(img_path)


if __name__ == "__main__":
    run()
