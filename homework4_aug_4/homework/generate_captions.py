import json
from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    # 1. Ego car
    # {kart_name} is the ego car.

    # 2. Counting
    # There are {num_karts} karts in the scenario.

    # 3. Track name
    # The track is {track_name}.

    # 4. Relative position
    # {kart_name} is {position} of the ego car.
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track = extract_track_info(info_path)

    ego = next((k for k in karts if k["is_center_kart"]), None)
    others = [k for k in karts if not k["is_center_kart"]]

    captions = []

    if ego:
        captions.append(f"{ego['kart_name']} is the ego car.")

    captions.append(f"There are {len(karts)} kart{'s' if len(karts) != 1 else ''} in the scenario.")
    captions.append(f"The track is {track}.")

    if ego:
        ego_cx = ego["center"][0]
        ego_cy = ego["center"][1]
        for k in others:
            kx, ky = k["center"]
            lr = "left" if kx < ego_cx else "right"
            fb = "front" if ky < ego_cy else "behind"
            captions.append(f"{k['kart_name']} is to the {lr} and {fb} of the ego car.")

    return captions


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


def generate(data_dir: str = "../data/train", output_file: str = "../data/train/balanced_captions.json"):
    """
    Iterate all *_info.json files in data_dir, generate captions for every view,
    and write results to output_file in the format CaptionDataset expects.
    """
    data_path = Path(data_dir)
    info_files = sorted(data_path.glob("*_info.json"))

    if not info_files:
        print(f"No *_info.json files found in {data_path.resolve()}")
        return

    all_captions = []

    for info_file in info_files:
        base_name = info_file.stem.replace("_info", "")

        with open(info_file) as f:
            info = json.load(f)
        num_views = len(info.get("detections", []))

        for view_index in range(num_views):
            image_matches = list(info_file.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))
            if not image_matches:
                continue

            rel_image = f"train/{image_matches[0].name}"
            captions = generate_caption(str(info_file), view_index)
            for caption in captions:
                all_captions.append({
                    "image_file": rel_image,
                    "caption": caption,
                })

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_captions, f, indent=2)

    print(f"Done - wrote {len(all_captions)} captions to {output_path.resolve()}")


"""
Usage:
   python -m homework.generate_captions check    --info_file data/valid/00000_info.json --view_index 0
   python -m homework.generate_captions generate [--data_dir ../data/train] [--output_file ../data/train/balanced_captions.json]
"""


def main():
    fire.Fire({"check": check_caption, "generate": generate})


if __name__ == "__main__":
    main()
