import sys
import pandas as pd
import numpy as np
from skimage import io

def tissue_cut(raw_mask_path, chip_df_path, output_path):
    # Load the tissue mask
    raw_mask = io.imread(raw_mask_path)
    print("Raw mask set up.")

    # Currently, you are not transposing the mask. If needed, uncomment the following line.
    # tissue_mask = np.transpose(raw_mask)
    tissue_mask = raw_mask

    # Load the chip data
    chip_df = pd.read_csv(chip_df_path, sep='\t', comment='#')
    print("Chip data set up.")

    # Filter chip data based on mask dimensions
    chip_df = chip_df[(chip_df['x'] < tissue_mask.shape[1]) & (chip_df['y'] < tissue_mask.shape[0])]

    # Check each 'x' and 'y' in tissue_mask if its value is 1
    valid_rows = chip_df.apply(lambda row: tissue_mask[int(row['y']), int(row['x'])] == 1, axis=1)
    print("Bin found.")

    # Filter chip_df to retain valid rows
    filtered_chip_df = chip_df[valid_rows]
    print("Bin filtered.")

    # Write the filtered data to a file
    filtered_chip_df.to_csv(output_path, sep='\t')
    print("Data written to file.")

def parse_args(args):
    arg_dict = {}
    try:
        for i in range(1, len(args), 2):
            arg_dict[args[i]] = args[i + 1]
    except IndexError:
        print("Error: argument values missing for some flags")
        sys.exit(1)
    return arg_dict

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python process_images_data.py --mask <raw_mask_path> --gem <chip_df_path> --output <output_path>")
        sys.exit(1)
    
    args = parse_args(sys.argv)
    raw_mask_path = args.get('--mask')
    chip_df_path = args.get('--gem')
    output_path = args.get('--output')

    if not all([raw_mask_path, chip_df_path, output_path]):
        print("Error: Missing one or more arguments.")
        sys.exit(1)

    tissue_cut(raw_mask_path, chip_df_path, output_path)