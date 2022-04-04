# ai4covid

In this version, there are checkpoints and also the data is included under the data folder. We include our preprocessing scripts as well. 

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

After this step the base and explain models can be run by running the following command:

```bash
python run_inference.py --chkpt_dir=data/checkpoints/ENSEMBLE/2022-03-02_15:27:01.765818 --test # base model
python run_xplain_inference.py --chkpt_dir=data/checkpoints_xplainable/2022-03-03_10:02:25.306729 --test # xplainable model
```

These ones are working examples. The base model has the output predictions in each folder in the checkpoints directory and the xplainable model has the output predictions in the xplainable_checkpoints directory. 

The ensemble prediction was created with the ensemble.ipynb notebook in the notebooks folder.

The best base model ensemble submission was the file `notebooks/mode_preds.csv`. The xplainable model submission was: `data/checkpoints_xplainable/2022-03-03_16:43:37.196679/pred_xplain.csv` (not exactly sure about this).
