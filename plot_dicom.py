# %%
import pydicom
from core.visualizer import show_dcm_info, plot_pixel_array

# %%

file_path = './dataset/sample.dicom'
dataset = pydicom.dcmread(file_path)
show_dcm_info(dataset)
plot_pixel_array(dataset)

# %%


