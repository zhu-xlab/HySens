from datasets.pavia import PaviaDataset, load_pavia_wavelengths, create_data_loaders_pavia
from datasets.indian_pines import IndianDataset, load_indian_wavelengths, create_data_loaders_indian
from datasets.whu_hc import WHUHC_Dataset, load_whuhc_wavelengths, create_data_loaders_whuhc
from datasets.whu_hh import WHUHH_Dataset, load_whuhh_wavelengths, create_data_loaders_whuhh
from datasets.hermiston import HermistonDataset, load_hermiston_wavelengths, create_data_loaders_hermiston
from datasets.bay import BayDataset, load_bay_wavelengths, create_data_loaders_bay
from datasets.barbara import BarbaraDataset, load_barbara_wavelengths, create_data_loaders_barbara
from datasets.augsburg import AugsburgDataset, load_augsburg_wavelengths, create_data_loaders_augsburg
from datasets.berlin import BerlinDataset, load_berlin_wavelengths, create_data_loaders_berlin
from datasets.gloria import GloriaDataset, load_gloria_wavelengths, create_data_loaders_gloria
from datasets.karly import KarlyDataset, load_karly_wavelengths, create_data_loaders_karly


def get_dataset(name):
    if name == "pavia":
        return PaviaDataset
    elif name == "indian_pines":
        return IndianDataset
    elif name == "whu_hc":
        return WHUHC_Dataset
    elif name == "whu_hh":
        return WHUHH_Dataset
    elif name == "hermiston":
        return HermistonDataset
    elif name == "bay":
        return BayDataset
    elif name == "barbara":
        return BarbaraDataset
    elif name == "augsburg":
        return AugsburgDataset
    elif name == "berlin":
        return BerlinDataset
    elif name == "gloria":
        return GloriaDataset
    elif name == "karly":
        return KarlyDataset
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_wavelength_loader(name):
    if name == "pavia":
        return load_pavia_wavelengths
    elif name == "indian_pines":
        return load_indian_wavelengths
    elif name == "whu_hc":
        return load_whuhc_wavelengths
    elif name == "whu_hh":
        return load_whuhh_wavelengths
    elif name == "hermiston":
        return load_hermiston_wavelengths
    elif name == "bay":
        return load_bay_wavelengths
    elif name == "barbara":
        return load_barbara_wavelengths
    elif name == "augsburg":
        return load_augsburg_wavelengths
    elif name == "berlin":
        return load_berlin_wavelengths
    elif name == "gloria":
        return load_gloria_wavelengths
    elif name == "karly":
        return load_karly_wavelengths
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_dataloader(name):
    if name == "pavia":
        return create_data_loaders_pavia
    elif name == "indian_pines":
        return create_data_loaders_indian
    elif name == "whu_hc":
        return create_data_loaders_whuhc
    elif name == "whu_hh":
        return create_data_loaders_whuhh
    elif name == "hermiston":
        return create_data_loaders_hermiston
    elif name == "bay":
        return create_data_loaders_bay
    elif name == "barbara":
        return create_data_loaders_barbara
    elif name == "augsburg":
        return create_data_loaders_augsburg
    elif name == "berlin":
        return create_data_loaders_berlin
    elif name == "gloria":
        return create_data_loaders_gloria
    elif name == "karly":
        return create_data_loaders_karly
    else:
        raise ValueError(f"Unknown dataset: {name}")