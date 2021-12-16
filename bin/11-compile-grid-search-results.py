if __name__ == "__main__":
    import os, pandas as pd, argparse, pickle

    ############### Parse commnd line args ###################
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_models',
        default="/content/output/mlp/500m_cloudthr80_lulcmndwi_masking_tmp_5fold",
        type=str,
        help="Path the the model outputs")
    args = parser.parse_args()

    directory = args.path_to_models
    
    paths = [f"{directory}/{x}" for x in os.listdir(directory)]

    metadata = pd.DataFrame(
        columns=[
            "id", "path", "features", "layers", "activation_layer", "epochs",
            "sched_gamma", "sched_step", "batch_size",
            "val_site_mse", "val_pooled_mse"
        ]
    )
    for path in paths:
        with open(path, "rb") as f:
            model_dict = pickle.load(f)

        row_dict = {
            "id": os.path.basename(path)[:-7],
            "path": path,
            "features": str(model_dict["features"]),
            "layers": model_dict["layer_out_neurons"],
            "activation_layer": model_dict["activation"],
            "epochs": model_dict["epochs"],
            "batch_size": model_dict["batch_size"],
            "sched_gamma": model_dict["learn_sched_gamma"],
            "sched_step": model_dict["learn_sched_step_size"],
            "val_site_mse": model_dict["val_site_mse"],
            "val_pooled_mse": model_dict["val_pooled_mse"]
        }

        metadata = metadata.append(row_dict, ignore_index=True)
    
    metadata.to_csv(f"output/mlp/grid_search_metadata.csv", index=False)
