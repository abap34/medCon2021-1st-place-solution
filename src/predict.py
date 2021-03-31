import utils
from models import wavenet, lstm, resnet_1, resnet_2

from omegaconf import OmegaConf
import numpy as np

MODEL_NAMES_DICT = {
    'wavenet': wavenet.Model,
    "resnet_1": resnet_1.Model,
    "resnet_2": resnet_2.Model,
    "lstm": lstm.Model
}

N_FOLD = 5


def main(param):
    utils.seed_everything(0)
    utils.info('read csv...')
    train, test, submit = utils.read_data("./data")
    utils.info('read wave data...')

    test_wave = utils.read_wave("./data/ecg/" + test["Id"] + ".npy")

    train["sex"] = train["sex"].replace({"male": 0, "female": 1})
    test["sex"] = test["sex"].replace({"male": 0, "female": 1})

    test_preds = np.zeros(
        [
            N_FOLD,
            test_wave.shape[0]
        ]
    )

    for fold in range(N_FOLD):
        utils.info('predict', fold)

        model = MODEL_NAMES_DICT[param.model_name](param)

        test_preds[fold] = model.predict([test_wave, test[["sex", "age"]]], fold)

    submit["target"] = test_preds.mean(axis=0)
    submit.to_csv("./logs/{}/submission.csv".format(param.model_name), index=False)


if __name__ == '__main__':
    param = OmegaConf.from_cli()
    utils.info('params:', param)
    main(param)
