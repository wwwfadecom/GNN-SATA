hp_cora = {
    "wd1": 4e-2,
    "wd2": 2e-5,
    "lambda_1": 4,
    "lambda_2": 0.01,
    "layer":3,
    "dropout": 0.35,
    "alpha": 0.001,
    "lr": 0.03,
}


hp_citeseer = {
    "wd1": 5e-2,
    "wd2": 2e-4,
    "lambda_1": 10,
    "lambda_2": 0.025,
    "layer": 3,
    "dropout": 0.15,
    "alpha": 0.001,
    "lr": 0.015,
}

hp_chameleon = {
    "wd1": 8e-2,
    "wd2": 0.0,
    "lambda_1": 7,
    "lambda_2": 0.005,
    "layer": 4,
    "dropout": 2e-2,
    "alpha": 0.004,
    "lr": 0.015,
}

hp_squirrel = {
    "wd1": 0.075,
    "wd2": 0.0,
    "lambda_1": 7,
    "lambda_2": 0.0002,
    "layer": 2,
    "dropout": 0.0,
    "alpha": 0.4,
    "lr": 0.005,
}

hp_amazon_computers = {
    "wd1": 5e-4,
    "wd2": 0.0,
    "lambda_1": 7,
    "lambda_2": 0.0001,
    "layer": 3,
    "dropout": 0.05,
    "alpha": 0.001,
    "lr": 0.03,
}

hp_amazon_photo = {
    "wd1": 5e-4,
    "wd2": 0.0,
    "lambda_1": 7,
    "lambda_2": 0.01,
    "layer": 2,
    "dropout": 0.05,
    "alpha": 0.001,
    "lr": 0.03,
}


def get_hyper_param(name: str):
    name = name.lower()
    if name == "cora":
        return hp_cora
    elif name == "citeseer":
        return hp_citeseer
    elif name == "chameleon":
        return hp_chameleon
    elif name == "squirrel":
        return hp_squirrel
    elif name == "computers":
        return hp_amazon_computers
    elif name == "photo":
        return hp_amazon_photo
    else:
        raise Exception("Not available")
