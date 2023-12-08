import os
from textwrap import shorten
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import tqdm
import shutil

if __name__ == "__main__":

    img_dir = "./GRAZPEDWRI-DX/data/images/"
    ann_dir = "./GRAZPEDWRI-DX/data/labels/"
    df = pd.read_csv("./GRAZPEDWRI-DX/dataset.csv")

    splitter1 = GroupShuffleSplit(test_size=.3, n_splits=2)
    split = splitter1.split(df, groups=df["patient_id"])
    train_idxs, valid_idxs = next(split)
    train_df = df.iloc[train_idxs]
    temp_df = df.iloc[valid_idxs]

    splitter2 = GroupShuffleSplit(test_size=.33333, n_splits=2)
    split = splitter2.split(temp_df, groups=temp_df["patient_id"])
    valid_idxs, test_idxs = next(split)
    valid_df = temp_df.iloc[valid_idxs]
    test_df = temp_df.iloc[test_idxs]

    train_df.to_csv("./GRAZPEDWRI-DX/train_data.csv", index=False)
    valid_df.to_csv("./GRAZPEDWRI-DX/valid_data.csv", index=False)
    test_df.to_csv("./GRAZPEDWRI-DX/test_data.csv", index=False)

    img_train_dir = "./GRAZPEDWRI-DX/data/images/train/"
    img_valid_dir = "./GRAZPEDWRI-DX/data/images/valid/"
    img_test_dir = "./GRAZPEDWRI-DX/data/images/test/"
    ann_train_dir = "./GRAZPEDWRI-DX/data/labels/train/"
    ann_valid_dir = "./GRAZPEDWRI-DX/data/labels/valid/"
    ann_test_dir = "./GRAZPEDWRI-DX/data/labels/test/"
    for dir in [img_train_dir, img_valid_dir, img_test_dir, ann_train_dir, ann_valid_dir, ann_test_dir]:
        if os.path.exists(dir) == False:
            os.makedirs(dir)

    for i in tqdm.tqdm(train_df.index, total=len(train_df)):
        filestem = train_df.loc[i, "filestem"]
        shutil.move(os.path.join(img_dir, filestem + ".png"), os.path.join(img_train_dir, filestem + ".png"))
        shutil.move(os.path.join(ann_dir, filestem + ".txt"), os.path.join(ann_train_dir, filestem + ".txt"))

    for i in tqdm.tqdm(valid_df.index, total=len(valid_df)):
        filestem = valid_df.loc[i, "filestem"]
        shutil.move(os.path.join(img_dir, filestem + ".png"), os.path.join(img_valid_dir, filestem + ".png"))
        shutil.move(os.path.join(ann_dir, filestem + ".txt"), os.path.join(ann_valid_dir, filestem + ".txt"))

    for i in tqdm.tqdm(test_df.index, total=len(test_df)):
        filestem = test_df.loc[i, "filestem"]
        shutil.move(os.path.join(img_dir, filestem + ".png"), os.path.join(img_test_dir, filestem + ".png"))
        shutil.move(os.path.join(ann_dir, filestem + ".txt"), os.path.join(ann_test_dir, filestem + ".txt"))
    
    N = len(df)
    print("Data split compleated according to PatientID:")
    print(f"  - {len(train_df)} ({100 * len(train_df)/N:.3f}%) images in the training set.")
    print(f"  - {len(valid_df)} ({100 * len(valid_df)/N:.3f}%) images in the validation set.")
    print(f"  - {len(test_df)} ({100 * len(test_df)/N:.3f}%) images in the testing set.")  
