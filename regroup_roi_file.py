from pathlib import Path
from shutil import copy
from random import shuffle


if __name__ == "__main__":
    LABEL_0_PATH = r"slapROIDataset\0"
    LABEL_1_PATH = r"slapROIDataset\1"
    label_0_img_list = list(Path(LABEL_0_PATH).glob("**/*.jpg"))
    label_1_img_list = list(Path(LABEL_1_PATH).glob("**/*.jpg"))
    shuffle(label_0_img_list)
    shuffle(label_1_img_list)
    label_0_split = int(len(label_0_img_list) * 0.1)
    label_1_split = int(len(label_1_img_list) * 0.1)
    train_path = Path(r"slapROIDataset\train")
    valid_path = Path(r"slapROIDataset\valid")
    for item in label_0_img_list[label_0_split:]:
        copy(item, train_path / "0" / "{}{}".format(item.stem, item.suffix))
    for item in label_1_img_list[label_1_split:]:
        copy(item, train_path / "1" / "{}{}".format(item.stem, item.suffix))
    for item in label_0_img_list[:label_0_split]:
        copy(item, valid_path / "0" / "{}{}".format(item.stem, item.suffix))
    for item in label_1_img_list[:label_1_split]:
        copy(item, valid_path / "1" / "{}{}".format(item.stem, item.suffix))