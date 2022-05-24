from .app import App
from .data import CIVETData
from .resources import config
from .utils import load_saved_model

import os

import pandas as pd

def main(argv=None):
    app = App()
    app.parse_args(argv=argv)
    data = CIVETData(app.args.filepath)
    model = load_saved_model(config['defaultModel'])
    predicted_qc = model.predict(data.features)
    df = pd.DataFrame({
        "ID": data.df[data.idvar],
        "QC": predicted_qc
    })
    df.to_csv(os.path.join(app.args.output_dir, "civetqc.csv"), index=False)


if __name__ == '__main__':
    main()