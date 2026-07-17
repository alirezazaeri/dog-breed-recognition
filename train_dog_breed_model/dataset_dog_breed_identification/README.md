# Dog Breed Identification Dataset

This project uses the public [Kaggle Dog Breed Identification dataset](https://www.kaggle.com/competitions/dog-breed-identification/data), containing metadata for 10,222 labelled images across 120 breed classes.

The image dataset is not included in this repository because of repository size and redistribution considerations. Review and accept the applicable Kaggle terms, download the data from the official source, and place the training images in the following local structure:

```text
dataset_dog_breed_identification/
├── labels.csv
└── main_dataset/
    └── <training JPG images>
```

`labels.csv` maps each image ID to its breed. The existing notebooks append the `.jpg` extension when matching metadata to files and expect the images at `dataset_dog_breed_identification/main_dataset/` when run from the `train_dog_breed_model/` directory.

The dataset and its images remain subject to the terms of their original source and are not covered by this repository's MIT License.
