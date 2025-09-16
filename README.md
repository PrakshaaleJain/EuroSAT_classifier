# EuroSAT RGB Dataset Splitter

This project provides tools for splitting the EuroSAT RGB dataset into training and validation sets for machine learning tasks.

## Project Structure

- `datset_splitter.py`: Script for splitting the dataset into train/val folders.
- `main.py`: Main entry point for running dataset operations.
- `EuroSAT_RGB/`: Contains the original EuroSAT RGB images organized by class.
- `EuroSAT_RGB_dataset/`: Contains the split dataset (`train/` and `val/` folders).

## Usage

1. Place the EuroSAT RGB dataset in the `EuroSAT_RGB/` folder.
2. Run `datset_splitter.py` to split the dataset into training and validation sets.
3. The split datasets will be available in `EuroSAT_RGB_dataset/train/` and `EuroSAT_RGB_dataset/val/`.

## Requirements

- Python 3.x
- Standard libraries (os, shutil, random, etc.)

## Example

```powershell
python datset_splitter.py
```

## License

This project is for educational purposes.
