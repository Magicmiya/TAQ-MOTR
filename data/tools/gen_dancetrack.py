import argparse
import os


def parse_option():
    # =================== Exp Configration ===================
    parser = argparse.ArgumentParser("The script of DanceTrack dataset processing", add_help=True)
    parser.add_argument("-P", "--data_path", type=str, help="dataset root path", required=True)

    return parser.parse_args()


def gen_dancetrack(data_root: str):
    data_set_path = os.path.join(data_root, "DanceTrack")
    if os.path.exists(data_set_path):
        files = os.listdir(data_set_path)
        for data_split in files:
            sub_path = os.path.join(data_set_path, data_split)
            if data_split in ["train", "val", "test"] and os.path.isdir(sub_path):
                seq_map_path = os.path.join(data_set_path, f'{data_split}_seqmap.txt')
                videos = sorted(os.listdir(sub_path))
                with open(seq_map_path, "w") as f:
                    for video in videos:
                        if video.startswith("dancetrack"):
                            f.write(video + '\n')
            else:
                continue
    else:
        raise FileNotFoundError(f"Do not find the DanceTrack dataset at {data_root}")


if __name__ == "__main__":
    print("\npreprocess of DanceTrack dataset")
    args = parse_option()
    gen_dancetrack(args.data_path)
    print("Successful finished")
