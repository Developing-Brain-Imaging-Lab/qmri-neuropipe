"""
I/O module for qMRI neuroimaging pipeline.

Provides data loading, BIDS dataset handling, and file I/O utilities.
"""
from qmri_neuropipe.io.data_loader import (
    DataLoader,
    SubjectData,
    DataTypeFiles,
    load_subject_data
)

from qmri_neuropipe.io.bids import (
    select_participants_sessions
)
from qmri_neuropipe.io.bids_inventory import (
    BIDSDatasetInventory,
    DataInventory,
    DerivativeProductCoverage,
    DerivativeInventory,
    ModelCoverage,
    ModalityCoverage,
    ProcessingGapCoverage,
    SessionBreakdown,
    inspect_bids_dataset,
)

__all__ = [
    'DataLoader',
    'SubjectData', 
    'DataTypeFiles',
    'load_subject_data',
    'select_participants_sessions',
    'BIDSDatasetInventory',
    'DataInventory',
    'DerivativeProductCoverage',
    'DerivativeInventory',
    'ModelCoverage',
    'ModalityCoverage',
    'ProcessingGapCoverage',
    'SessionBreakdown',
    'inspect_bids_dataset',
]
