import json
import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import GCPDataset, load_test_samples, val_transform, SHAPE_CLASSES
from model import GCPModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_DIR = "test_dataset"
WEIGHTS_PATH = "weights/best_model.pth"
OUTPUT_JSON = "predictions.json"


def run_inference():

    model = GCPModel().to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH,map_location=device))
    model.eval()

    test_samples = load_test_samples(TEST_DIR)

    test_ds = GCPDataset(test_samples,TEST_DIR,val_transform)

    test_loader = DataLoader(test_ds,batch_size=8)

    predictions = {}

    with torch.no_grad():

        for imgs,_,_,rel_paths,orig_ws,orig_hs in test_loader:

            imgs = imgs.to(device)

            pc,pl = model(imgs)

            pc = torch.sigmoid(pc).cpu().numpy()

            for i,rel in enumerate(rel_paths):

                x = float(pc[i,0]*orig_ws[i])
                y = float(pc[i,1]*orig_hs[i])

                shape = SHAPE_CLASSES[pl[i].argmax().item()]

                predictions[rel] = {
                    "mark":{"x":x,"y":y},
                    "verified_shape":shape
                }

    with open(OUTPUT_JSON,"w") as f:
        json.dump(predictions,f,indent=2)

    print("Predictions saved")


if __name__ == "__main__":
    run_inference()
