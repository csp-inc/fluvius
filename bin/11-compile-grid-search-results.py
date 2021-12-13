if __name__ == "__main__":
    import os, sys, pandas as pd, numpy as np, argparse, pickle

    ############### Parse commnd line args ###################
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_models',
        default="output/mlp/500m_cloudthr80_lulcmndwi_masking",
        type=str,
        help="Path the the model outputs")
    args = parser.parse_args()

    directory = args.path_to_models
    
    paths = [f"{directory}/{x}" for x in os.listdir(directory)]

    metadata = pd.DataFrame(
        columns=[
            "id", "path", "features", "epochs", "learning_rate", "batch_size",
            "val_loss", "val_R2", "itv_R2"
        ]
    )
    for path in paths:
        with open(path, "rb") as f:
            model_dict = pickle.load(f)

        row_dict = {
            "id": os.path.basename(path)[:-7],
            "path": path,
            "features": str(model_dict["features"]),
            "epochs": model_dict["epochs"],
            "learning_rate": model_dict["learning_rate"],
            "batch_size": model_dict["batch_size"],
            "train_loss": model_dict["loss_stats"]["train"][-1],
            "val_loss": model_dict["loss_stats"]["val"][-1],
            "val_R2": model_dict["val_R2"],
            "itv_R2": model_dict["itv_R2"]
        }

        metadata = metadata.append(row_dict, ignore_index=True)
    
    metadata.to_csv(f"{directory}/grid_search_metadata.csv", index=False)


    