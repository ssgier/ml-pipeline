from ml_pipeline.util import make_convolution_weight_mask, make_proximity_weight_mask
import matplotlib.pyplot as plt
import numpy as np

m = make_proximity_weight_mask(4, 1)

ms = [subm.reshape((4, 4)) for subm in m]
print(len(ms))

fig, axs = plt.subplots(4, 4, figsize=(15, 6))  # 2 rows, 5 columns

# Flatten the axs array for easy iteration if it's multidimensional
axs = axs.flatten()

# Loop through the list of arrays and their corresponding axes to display each array
for i, ax in enumerate(axs):
    ax.imshow(ms[i], cmap="gray")
    ax.set_title(i)  # Optional: add title to each subplot
    ax.axis("off")  # Optional: hide axes for cleaner visualization

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()
