import argparse
import os
import zipfile


SAVE_FORMAT = "{frame},{id},{x1:.3f},{y1:.3f},{w:.2f},{h:.2f},{s:.2f},-1,-1,-1\n"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert existing SportsMOT tracking results to 1-based submission format."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing original SportsMOT result txt files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to store corrected txt files.",
    )
    parser.add_argument(
        "--zip-path",
        required=True,
        help="Path of the final submission zip file.",
    )
    return parser.parse_args()


def iter_txt_files(input_dir: str):
    for file_name in sorted(os.listdir(input_dir)):
        if file_name.endswith(".txt"):
            yield file_name


def convert_line(line: str) -> str:
    cols = [item.strip() for item in line.strip().split(",")]
    if len(cols) != 10:
        raise ValueError(f"Expected 10 columns, got {len(cols)}: {line.rstrip()}")

    frame = int(float(cols[0]))
    track_id = int(float(cols[1])) + 1
    x1 = float(cols[2]) + 1.0
    y1 = float(cols[3]) + 1.0
    w = float(cols[4])
    h = float(cols[5])
    score = float(cols[6])
    return SAVE_FORMAT.format(frame=frame, id=track_id, x1=x1, y1=y1, w=w, h=h, s=score)


def convert_dir(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    file_count = 0
    row_count = 0

    for file_name in iter_txt_files(input_dir):
        src_path = os.path.join(input_dir, file_name)
        dst_path = os.path.join(output_dir, file_name)
        with open(src_path, "r", encoding="utf-8") as src, open(dst_path, "w", encoding="utf-8") as dst:
            for row in src:
                stripped = row.strip()
                if not stripped:
                    continue
                dst.write(convert_line(stripped))
                row_count += 1
        file_count += 1

    return file_count, row_count


def build_zip(result_dir: str, zip_path: str):
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for file_name in iter_txt_files(result_dir):
            zip_file.write(os.path.join(result_dir, file_name), arcname=file_name)


def main():
    args = parse_args()
    file_count, row_count = convert_dir(input_dir=args.input_dir, output_dir=args.output_dir)
    build_zip(result_dir=args.output_dir, zip_path=args.zip_path)
    print(f"fixed_files={file_count}")
    print(f"fixed_rows={row_count}")
    print(f"output_dir={args.output_dir}")
    print(f"zip_path={args.zip_path}")


if __name__ == "__main__":
    main()
