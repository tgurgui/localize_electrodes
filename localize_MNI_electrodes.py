import numpy as np
import nibabel as nib
import pandas as pd

def region_points(volume_filename, points):
	vol_nii = nib.load(volume_filename)
	vol_data = vol_nii.get_fdata()
	vol_affine = vol_nii.affine

	regions = []
	for point in points:
		pinv = np.linalg.inv(vol_affine).dot([point[0], point[1],point[2],1])[:-1].astype(int)
		regions.append(vol_data[pinv[0], pinv[1], pinv[2]])

	return regions


if __name__ == "__main__":
	aseg_filename = "aseg_MNI.mgz"
	electrodes_filename = "channel_MNI.txt"

	points_df = pd.read_csv(electrodes_filename, sep='\t', index_col=None, names=['label', 'x', 'y', 'z'])

	points = points_df[['x', 'y', 'z']]
	regions = region_points(aseg_filename, points.to_numpy())
	points_df['area'] = ''
	points_df['area'] = regions
	points_df = points_df[['label', 'x', 'y', 'z', 'area']]
	points_df.to_csv(electrodes_filename, sep="\t", index=None)
