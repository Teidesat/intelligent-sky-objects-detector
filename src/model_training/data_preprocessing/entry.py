import numpy as np


class DatasetEntry:
    """Struct with the processed data for a single dataset entry."""

    def __init__( self, entry_id: str, nn_input_image: np.ndarray, 
                 segmentation_mask: np.ndarray, filtered_objects: list, ):
        self.entry_id = entry_id
        self.nn_input_image = nn_input_image       
        self.segmentation_mask = segmentation_mask 
        self.filtered_objects = filtered_objects    