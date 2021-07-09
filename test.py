import pathlib
import sys
from collections import defaultdict
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import DatasetImageCoord
from models import Posxymodel
import h5py
from tqdm import tqdm
import pandas as pd


def save_reconstructions(reconstructions, out_dir):

    df = pd.DataFrame(reconstructions)
    df.to_csv(out_dir / "result.csv")


def create_data_loaders(args):

    data = DatasetImageCoord(args.data_path, args.eval_csv_path, args.pos_csv_path)

    data_loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
    )

    return data_loader


def load_model(checkpoint_file):

    checkpoint = torch.load(checkpoint_file)
    model = Posxymodel().to(args.device)
    model.load_state_dict(checkpoint["model"])

    return model


def run_unet(args, model, data_loader):
    model.eval()
    reconstructions = {"fname": [], "x": [], "y": []}
    with torch.no_grad():
        for (iter, data) in enumerate(tqdm(data_loader)):

            input, target, dim, fnames = data

            input = input.to(args.device)
            target = target.to(args.device)

            input = input.float()
            target = target.float()

            recons = model(input.float())

            #print (recons.shape, dim.shape)
            #print (recons, dim)

            recons = recons.to("cpu").squeeze(1)

            for i in range(recons.shape[0]):
                reconstructions["fname"].append(fnames[i])
                reconstructions["x"].append(recons[i].numpy()[0] * dim[i].numpy()[1])
                reconstructions["y"].append(recons[i].numpy()[1] * dim[i].numpy()[0])

    return reconstructions


def main(args):

    data_loader = create_data_loaders(args)
    model = load_model(args.checkpoint)

    reconstructions = run_unet(args, model, data_loader)

    save_reconstructions(reconstructions, args.out_dir)


def create_arg_parser():

    parser = argparse.ArgumentParser(description="Valid setup for MR recon U-Net")
    parser.add_argument(
        "--checkpoint", type=pathlib.Path, required=True, help="Path to the U-Net model"
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        required=True,
        help="Path to save the reconstructions to",
    )
    parser.add_argument("--batch-size", default=16, type=int, help="Mini-batch size")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Which device to run on"
    )
    parser.add_argument("--data-path", type=str, help="path to validation dataset")
    parser.add_argument("--eval-csv-path", type=str, help="Path to test h5 files")
    parser.add_argument("--pos-csv-path", type=str, help="Path to test h5 files")
    return parser


if __name__ == "__main__":
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
