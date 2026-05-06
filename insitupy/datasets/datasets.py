import hashlib
import logging
import os
import shutil
from pathlib import Path

from insitupy._constants import CACHE

logger = logging.getLogger(__name__)
from insitupy._core.data import InSituData
from insitupy.datasets.download import download_url
from insitupy.io.data import read_visium, read_xenium

# parameters for download functions
DEMODIR = CACHE / 'demo_datasets'

# helper function to resolve output directory
def _resolve_output_dir(output_dir):
    """
    Resolve the output directory for datasets.

    Parameters
    ----------
    output_dir : str, Path, or None
        Custom output directory. If None, uses default DEMODIR.

    Returns
    -------
    Path
        Resolved output directory path.
    """
    if output_dir is None:
        return DEMODIR
    else:
        output_path = Path(output_dir)
        if not output_path.exists():
            raise ValueError(f"The specified output directory does not exist: {output_dir}")
        return output_path

# functions that each download a dataset into  '~/.cache/InSituPy/demo_dataset'
def _md5sum(filePath):
    with open(filePath, 'rb') as fh:
        m = hashlib.md5()
        while True:
            data = fh.read(8192)
            if not data:
                break
            m.update(data)
    return m.hexdigest()

# function that checks the md5sum of the images and returns a boolean.
def _md5sum_image_check(file_path : Path, expected_md5sum, overwrite):
    download = False
    if file_path.exists():
        logger.info("Image exists. Checking md5sum...")
        if _md5sum(file_path) == expected_md5sum:
            if not overwrite:
                logger.info("The md5sum matches. Download is skipped. To force download set `overwrite=True`.")
                return
            else:
                # if the md5sum matches but overwrite=True, the image is still downloaded
                download = True
        else:
            logger.info("The md5sum doesn't match. Image is downloaded.")
            download = True
    else:
        download = True

    return download


def list_downloaded_datasets(output_dir=None):
    """
    List all downloaded demo datasets in a directory.

    Parameters
    ----------
    output_dir : str, Path, or None
        Directory to check for datasets. If None, uses default DEMODIR.
    """
    check_dir = _resolve_output_dir(output_dir) if output_dir is not None else DEMODIR

    try:
        # List all items in the given directory
        items = os.listdir(check_dir)
        # Filter out only the folders
        folders = [item for item in items if os.path.isdir(os.path.join(check_dir, item))]

        if folders:
            logger.info(f"Following demo datasets were found in {check_dir}:\n")
            for folder in folders:
                logger.info(f"- {folder}")
        else:
            logger.info(f"No folders found in the directory: {check_dir}")
    except FileNotFoundError:
        logger.warning(f"The directory '{check_dir}' does not exist.")
    except PermissionError:
        logger.warning(f"Permission denied to access the directory '{check_dir}'.")


# function that checks data for md5sum, downloads and unpacks the data.
def _data_check_and_download(data_dir, zip_file, expected_md5sum, overwrite, data_url, named_data_dir):
    # check if the unzipped data exists
    download_data = False
    if data_dir.exists():
        # if it exists, everything is fine and it is assumed that the dataset was downloaded correctly. Overwrite is still checked.
        if not overwrite:
            logger.info("This dataset exists already. Download is skipped. To force download set `overwrite=True`.")
            return
        else:
            logger.info("This dataset exists already but is overwritten because of `overwrite=True`.")
            download_data = True
    else:
        # if unzipped data does not exist, we need to check if a zip file exists, and if yes if its md5sum is correct
        if zip_file.exists():
            logger.info("ZIP file exists. Checking md5sum...")
            if _md5sum(zip_file) == expected_md5sum:
                if not overwrite:
                    logger.info("This dataset exists already. Download is skipped. To force download set `overwrite=True`.")
                    return
                else:
                    # if the md5sum matches but overwrite=True, the data is still downloaded
                    download_data = True
            else:
                logger.warning("The dataset exists already but the md5sum is not as expected. Dataset is downloaded again.")
                download_data = True
        else:
            download_data = True

    if download_data:
        # download data as zip file
        download_url(data_url, out_dir=named_data_dir, overwrite=True)

        # unzip data
        shutil.unpack_archive(zip_file, data_dir)

        # move files from outs folder into parent directory and remove the outs folder
        if (data_dir / "outs/").exists():
            for f in data_dir.glob("outs/*"):
                shutil.move(f, data_dir)
            os.rmdir(data_dir / "outs")

        #remove zip file after unpacking
        os.remove(zip_file)


# Xenium onboard analysis version 1.0.1
# data from https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast
def xenium_human_breast_cancer(
        overwrite: bool = False,
        output_dir = None
) -> InSituData:
    """Load human FFPE breast cancer Xenium data (v1.0.1, ~160k cells, 10x Genomics preview dataset)."""

    # URLs for download
    data_url = "https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_outs.zip"
    he_url = "https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.ome.tif"
    if_url = "https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_if_image.ome.tif"

    # set up paths
    base_dir = _resolve_output_dir(output_dir)
    named_data_dir = base_dir / "xenium_hbreastcancer"
    data_dir = named_data_dir / "output-XETG00000__slide_id__hbreastcancer"
    image_dir = named_data_dir / "unregistered_images"
    zip_file = named_data_dir / Path(data_url).name

    # check if file exists and has correct md5sum
    expected_md5sum = '7d42a0b232f92a2e51de1f513b1a44fd'
    expected_he_md5sum = 'fc0d0d38b7c039cc0682e51099f8d841'
    expected_if_md5sum = '929839c64ef8331cfd048a614f5f6829'

    # image file names
    he_file_name = "slide_id__hbreastcancer__HE__histo"
    if_file_name = "slide_id__hbreastcancer__CD20_HER2_DAPI__IF"
    # check if data exists (zipped or unzipped), if yes check md5sum
    # if necessary download data
    _data_check_and_download(data_dir, zip_file, expected_md5sum, overwrite, data_url, named_data_dir)

    # download image data
    if _md5sum_image_check(image_dir/"slide_id__hbreastcancer__HE__histo.ome.tif", expected_he_md5sum, overwrite):
        download_url(he_url, out_dir = image_dir, file_name = he_file_name, overwrite = True)

    if _md5sum_image_check(image_dir/"slide_id__hbreastcancer__CD20_HER2_DAPI__IF.ome.tif", expected_if_md5sum, overwrite):
        download_url(if_url, out_dir = image_dir, file_name = if_file_name, overwrite = True)

    logger.info(f"Corresponding image data can be found in {image_dir}")
    logger.info("For this dataset following images are available:")
    logger.info(f"{he_file_name}.ome.tiff")
    logger.info(f"{if_file_name}.ome.tiff")

    # load data into InSituData object
    data = read_xenium(data_dir)

    return data

# xenium onboard analysis version 1.5.0
# data from https://www.10xgenomics.com/resources/datasets/human-kidney-preview-data-xenium-human-multi-tissue-and-cancer-panel-1-standard
def xenium_human_kidney_nondiseased(
        overwrite: bool = False,
        output_dir = None
) -> InSituData:
    """Load human non-diseased FFPE kidney Xenium data (v1.5.0, 10x Genomics multi-tissue and cancer panel)."""

    # URLs for download
    data_url = "https://cf.10xgenomics.com/samples/xenium/1.5.0/Xenium_V1_hKidney_nondiseased_section/Xenium_V1_hKidney_nondiseased_section_outs.zip"
    he_url = "https://cf.10xgenomics.com/samples/xenium/1.5.0/Xenium_V1_hKidney_nondiseased_section/Xenium_V1_hKidney_nondiseased_section_he_image.ome.tif"

    # set up paths
    base_dir = _resolve_output_dir(output_dir)
    named_data_dir = base_dir / "xenium_hkidney"
    data_dir = named_data_dir / "output-XETG0000__slide_id__hkidney"
    image_dir = named_data_dir / "unregistered_images"
    zip_file = named_data_dir / Path(data_url).name

    # check if file exists and has correct md5sum
    expected_md5sum = '194d5e21b40b27fa8c009d4cbdc3272d'
    expected_he_md5sum = 'e457889aea78bef43834e675f0c58d95'

    # image file name
    he_file_name = "slide_id__hkidney__HE__histo"

    # check if data exists (zipped or unzipped), if yes check md5sum
    # if necessary download data
    _data_check_and_download(data_dir, zip_file, expected_md5sum, overwrite, data_url, named_data_dir)

    # download image data
    if _md5sum_image_check(image_dir/"slide_id__hkidney__HE__histo.ome.tif", expected_he_md5sum, overwrite):
        download_url(he_url, out_dir = image_dir, file_name = he_file_name, overwrite = True)

    logger.info(f"Corresponding image data can be found in {image_dir}")
    logger.info("For this dataset following image is available:")
    logger.info(f"{he_file_name}.ome.tiff")

    # load data into InSituData object
    data = read_xenium(data_dir)

    return data

# xenium onboard analysis version 1.6.0
# data from https://www.10xgenomics.com/datasets/pancreatic-cancer-with-xenium-human-multi-tissue-and-cancer-panel-1-standard
def xenium_human_pancreatic_cancer(
        overwrite: bool = False,
        output_dir = None
) -> InSituData:
    """Load human FFPE pancreatic cancer Xenium data (v1.6.0, add-on panel, 10x Genomics)."""

    # URLs for download
    data_url = "https://cf.10xgenomics.com/samples/xenium/1.6.0/Xenium_V1_hPancreas_Cancer_Add_on_FFPE/Xenium_V1_hPancreas_Cancer_Add_on_FFPE_outs.zip"
    he_url = "https://cf.10xgenomics.com/samples/xenium/1.6.0/Xenium_V1_hPancreas_Cancer_Add_on_FFPE/Xenium_V1_hPancreas_Cancer_Add_on_FFPE_he_image.ome.tif"
    if_url = "https://cf.10xgenomics.com/samples/xenium/1.6.0/Xenium_V1_hPancreas_Cancer_Add_on_FFPE/Xenium_V1_hPancreas_Cancer_Add_on_FFPE_if_image.ome.tif"

    # set up paths
    base_dir = _resolve_output_dir(output_dir)
    named_data_dir = base_dir / "xenium_hpancreas"
    data_dir = named_data_dir / "output-XETG00000__slide_id__hpancreas"
    image_dir = named_data_dir / "unregistered_images"
    zip_file = named_data_dir / Path(data_url).name

    # check if file exists and has correct md5sum
    expected_md5sum = '7acca4c2a40f09968b72275403c29f93'
    expected_he_md5sum = '4e96596ea13a3d0f6139638b2b90aef4'
    expected_if_md5sum = 'c859a7ab5d29807b4daf1f66cb6f5060'

    # image file names
    he_file_name = "slide_id__hPancreas__HE__histo"
    if_file_name = "slide_id__hPancreas__CD20_TROP2_PPY_DAPI__IF"

    # check if data exists (zipped or unzipped), if yes check md5sum
    # if necessary download data)
    _data_check_and_download(data_dir, zip_file, expected_md5sum, overwrite, data_url, named_data_dir)

    # download image data
    if _md5sum_image_check(image_dir/"slide_id__hPancreas__HE__histo.ome.tif", expected_he_md5sum, overwrite):
        download_url(he_url, out_dir = image_dir, file_name = he_file_name, overwrite = True)

    if _md5sum_image_check(image_dir/"slide_id__hPancreas__CD20_TROP2_PPY_DAPI__IF.ome.tif", expected_if_md5sum, overwrite):
        download_url(if_url, out_dir = image_dir, file_name = if_file_name, overwrite = True )

    logger.info(f"Corresponding image data can be found in {image_dir}")
    logger.info("For this dataset following images are available:")
    logger.info(f"{he_file_name}.ome.tiff")
    logger.info(f"{if_file_name}IF_image_name.ome.tiff")

    # load data into InSituData object
    data = read_xenium(data_dir)

    return data

# xenium onboard analysis version 1.7.0
# data from https://www.10xgenomics.com/resources/datasets/human-skin-preview-data-xenium-human-skin-gene-expression-panel-add-on-1-standard
def xenium_human_skin_melanoma(
        overwrite: bool = False,
        output_dir = None
) -> InSituData:
    """Load human FFPE skin melanoma Xenium data (v1.7.0, skin gene expression panel, 10x Genomics)."""

    # URLs for download
    data_url = "https://cf.10xgenomics.com/samples/xenium/1.7.0/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE_outs.zip"
    he_url = "https://cf.10xgenomics.com/samples/xenium/1.7.0/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE_he_image.ome.tif"

    # set up paths
    base_dir = _resolve_output_dir(output_dir)
    named_data_dir = base_dir / "xenium_hskin"
    data_dir = named_data_dir / "output-XETG00000__slide_id__hskin"
    image_dir = named_data_dir / "unregistered_images"
    zip_file = named_data_dir / Path(data_url).name

    # check if file exists and has correct md5sum
    expected_md5sum = '29102799a3f1858c7318b705eb1a8584'
    expected_he_md5sum = '169af7630e0124eef61d252183243a06'

    # image file name
    he_file_name = "slide_id__hskin__HE__histo"

    # check if data exists (zipped or unzipped), if yes check md5sum
    # if necessary download data
    _data_check_and_download(data_dir, zip_file, expected_md5sum, overwrite, data_url, named_data_dir)

    # download image data
    if _md5sum_image_check(image_dir/"slide_id__hskin__HE__histo.ome.tif", expected_he_md5sum, overwrite):
        download_url(he_url, out_dir = image_dir, file_name = he_file_name, overwrite = True)

    logger.info(f"Corresponding image data can be found in {image_dir}")
    logger.info("For this dataset following image is available:")
    logger.info(f"{he_file_name}.ome.tiff")

    # load data into InSituData object
    data = read_xenium(data_dir)

    return data

# xenium onboard analysis version 2.0.0
# data from https://www.10xgenomics.com/datasets/ffpe-human-brain-cancer-data-with-human-immuno-oncology-profiling-panel-and-custom-add-on-1-standard
def xenium_human_brain_cancer(
        overwrite: bool = False,
        output_dir = None
) -> InSituData:
    """Load human FFPE GBM brain cancer Xenium data (v2.0.0, immuno-oncology profiling panel, 10x Genomics)."""

    # URLs for download
    data_url = "https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/2.0.0/Xenium_V1_Human_Brain_GBM_FFPE/Xenium_V1_Human_Brain_GBM_FFPE_outs.zip"
    he_url = "https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_Human_Brain_GBM_FFPE/Xenium_V1_Human_Brain_GBM_FFPE_he_image.ome.tif"

    # set up paths
    base_dir = _resolve_output_dir(output_dir)
    named_data_dir = base_dir / "xenium_hbraincancer"
    data_dir = named_data_dir / "output-XETG00000__slide_id__hbraincancer"
    image_dir = named_data_dir / "unregistered_images"
    zip_file = named_data_dir / Path(data_url).name

    # check if file exists and has correct md5sum
    expected_md5sum = "c116017ad9884cf6944c6c4815bffb3c"
    expected_he_md5sum = "22b66c6e7669933e50a9665d467e639f"

    # image file name
    he_file_name = "slide_id__hbraincancer__HE__histo"

    # check if data exists (zipped or unzipped), if yes check md5sum
    # if necessary download data
    _data_check_and_download(data_dir, zip_file, expected_md5sum, overwrite, data_url, named_data_dir)

    # download image data
    if _md5sum_image_check(image_dir/"slide_id__hbraincancer__HE__histo.ome.tif", expected_he_md5sum, overwrite):
        download_url(he_url, out_dir = image_dir, file_name = he_file_name, overwrite = True)

    logger.info(f"Corresponding image data can be found in {image_dir}")
    logger.info("For this dataset following image is available:")
    logger.info(f"{he_file_name}.ome.tiff")

    # load data into InSituData object
    data = read_xenium(data_dir)

    return data

# xenium onboard analysis 2.0.0
# data from https://www.10xgenomics.com/datasets/preview-data-ffpe-human-lung-cancer-with-xenium-multimodal-cell-segmentation-1-standard
def xenium_human_lung_cancer(
        overwrite: bool = False,
        output_dir = None
) -> InSituData:
    """Load human FFPE lung cancer Xenium data (v2.0.0, multimodal cell segmentation, 10x Genomics)."""

    # URLs for download
    data_url = "https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_humanLung_Cancer_FFPE/Xenium_V1_humanLung_Cancer_FFPE_outs.zip"
    he_url = "https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_humanLung_Cancer_FFPE/Xenium_V1_humanLung_Cancer_FFPE_he_image.ome.tif"

    # set up paths
    base_dir = _resolve_output_dir(output_dir)
    named_data_dir = base_dir / "xenium_hlungcancer"
    data_dir = named_data_dir / "output-XETG00000__slide_id__hlungcancer"
    image_dir = named_data_dir / "unregistered_images"
    zip_file = named_data_dir / Path(data_url).name

    # check if file exists and has correct md5sum
    expected_md5sum = "194e24c1efe7e64d2487adfe313bb9dd"
    expected_he_md5sum = "47147933d73e008a0dd3695895832dd4"

    # image file names
    he_file_name = "slide_id__hlungcancer__HE__histo"

    # check if data exists (zipped or unzipped), if yes check md5sum
    # if necessary download data
    _data_check_and_download(data_dir, zip_file, expected_md5sum, overwrite, data_url, named_data_dir)

    # download image data
    if _md5sum_image_check(image_dir/"slide_id__hlungcancer__HE__histo.ome.tif", expected_he_md5sum, overwrite):
        download_url(he_url, out_dir = image_dir, file_name = he_file_name, overwrite = True)

    logger.info(f"Corresponding image data can be found in {image_dir}")
    logger.info("For this dataset following image is available:")
    logger.info(f"{he_file_name}.ome.tiff")

    # load data into InSituData object
    data = read_xenium(data_dir)

    return data

# xenium onboard analysis 3.0.0
# data from https://www.10xgenomics.com/datasets/preview-data-xenium-prime-gene-expression
def xenium_human_lymph_node_5k(
        overwrite: bool = False,
        output_dir = None
) -> InSituData:
    """Load human reactive FFPE lymph node Xenium Prime data (v3.0.0, 5k gene panel, 10x Genomics)."""

    # URLs for download
    data_url = "https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/3.0.0/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_outs.zip"
    he_url = "https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_he_image.ome.tif"

    # set up paths
    base_dir = _resolve_output_dir(output_dir)
    named_data_dir = base_dir / "xenium_hlymphnode_5k"
    data_dir = named_data_dir / "output-XETG00000__slide_id__hlymphnode5k"
    image_dir = named_data_dir / "unregistered_images"
    zip_file = named_data_dir / Path(data_url).name

    # check if file exists and has correct md5sum
    expected_md5sum = "1ddb7d10e2bca93a61830d5dc57cb3a8"
    expected_he_md5sum = "eae078b0b3ddcaf6a11ca4eefabcaf0c"

    # image file names
    he_file_name = "slide_id__hlymphnode5k__HE__histo"

    # check if data exists (zipped or unzipped), if yes check md5sum
    # if necessary download data
    _data_check_and_download(data_dir, zip_file, expected_md5sum, overwrite, data_url, named_data_dir)

    # download image data
    if _md5sum_image_check(image_dir/"slide_id__hlymphnode5k__HE__histo.ome.tif", expected_he_md5sum, overwrite):
        download_url(he_url, out_dir = image_dir, file_name = he_file_name, overwrite = True)

    logger.info(f"Corresponding image data can be found in {image_dir}")
    logger.info("For this dataset following image is available:")
    logger.info(f"{he_file_name}.ome.tiff")

    # load data into InSituData object
    data = read_xenium(data_dir)

    return data

# xenium onboard analysis 1.5.0
# data from https://www.10xgenomics.com/datasets/human-lymph-node-preview-data-xenium-human-multi-tissue-and-cancer-panel-1-standard
def xenium_human_lymph_node(
        overwrite: bool = False,
        output_dir = None
) -> InSituData:
    """Load human non-diseased FFPE lymph node Xenium data (v1.5.0, multi-tissue and cancer panel, 10x Genomics)."""

    # URLs for download
    data_url = "https://cf.10xgenomics.com/samples/xenium/1.5.0/Xenium_V1_hLymphNode_nondiseased_section/Xenium_V1_hLymphNode_nondiseased_section_outs.zip"

    # set up paths
    base_dir = _resolve_output_dir(output_dir)
    named_data_dir = base_dir / "xenium_hlymphnode"
    data_dir = named_data_dir / "output-XETG00000__slide_id__hlymphnode"
    image_dir = named_data_dir / "unregistered_images"
    zip_file = named_data_dir / Path(data_url).name

    # check if file exists and has correct md5sum
    expected_md5sum = "4e9ef0a40b0fc00e619cf310f349f1bd"

    # check if data exists (zipped or unzipped), if yes check md5sum
    # if necessary download data
    _data_check_and_download(data_dir, zip_file, expected_md5sum, overwrite, data_url, named_data_dir)

    logger.info("For this dataset no additional images are available.")

    # load data into InSituData object
    data = read_xenium(data_dir)

    return data


# xenium onboard analysis 2.0.0
# data from https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_human_Breast_2fov/Xenium_V1_human_Breast_2fov_outs.zip
# Human breast, multimodal cell segmentation
def xenium_test_dataset_v2_mm(
        overwrite: bool = False,
        output_dir = None
) -> InSituData:
    """Load human FFPE breast cancer 2-FOV Xenium test data (v2.0.0, multimodal cell segmentation, 10x Genomics)."""
    data_url = "https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_human_Breast_2fov/Xenium_V1_human_Breast_2fov_outs.zip"
    base_dir = _resolve_output_dir(output_dir)
    named_data_dir = base_dir / "xenium_test_dataset_v2_mm"
    data_dir = named_data_dir / "output-XETG00000__slide_id__xenium_test_dataset_v2_mm"
    zip_file = named_data_dir / Path(data_url).name
    expected_md5sum = "4632914eca973a1d532231ea646e10cc"

    _data_check_and_download(data_dir, zip_file, expected_md5sum, overwrite, data_url, named_data_dir)
    data = read_xenium(data_dir)
    logger.info("For this dataset no additional images are available.")
    return data


# xenium onboard analysis 2.0.0
# data from https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_human_Lung_2fov/Xenium_V1_human_Lung_2fov_outs.zip
# Human lung, nuclear expansion
def xenium_test_dataset_v2_nucex(
        overwrite: bool = False,
        output_dir = None
) -> InSituData:
    """Load human FFPE lung 2-FOV Xenium test data (v2.0.0, nuclear expansion segmentation, 10x Genomics)."""
    data_url = "https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_human_Lung_2fov/Xenium_V1_human_Lung_2fov_outs.zip"
    base_dir = _resolve_output_dir(output_dir)
    named_data_dir = base_dir / "xenium_test_dataset_v2_nucex"
    data_dir = named_data_dir / "output-XETG00000__slide_id__xenium_test_dataset_v2_nucex"
    zip_file = named_data_dir / Path(data_url).name
    expected_md5sum = "bf9c5e6762681b81eab0a19d3d590381"

    _data_check_and_download(data_dir, zip_file, expected_md5sum, overwrite, data_url, named_data_dir)
    data = read_xenium(data_dir)
    logger.info("For this dataset no additional images are available.")
    return data


# xenium onboard analysis 3.0.0
# data from https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_MultiCellSeg_Mouse_Ileum_tiny/Xenium_Prime_MultiCellSeg_Mouse_Ileum_tiny_outs.zip
# Mouse ileum, multimodal cell segmentation
def xenium_test_dataset_v3_mm(
        overwrite: bool = False,
        output_dir = None
) -> InSituData:
    """Load mouse ileum tiny Xenium Prime test data (v3.0.0, multimodal cell segmentation, 10x Genomics)."""
    data_url = "https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_MultiCellSeg_Mouse_Ileum_tiny/Xenium_Prime_MultiCellSeg_Mouse_Ileum_tiny_outs.zip"
    base_dir = _resolve_output_dir(output_dir)
    named_data_dir = base_dir / "xenium_test_dataset_v3_mm"
    data_dir = named_data_dir / "output-XETG00000__slide_id__xenium_test_dataset_v3_mm"
    zip_file = named_data_dir / Path(data_url).name
    expected_md5sum = "be9d917eaac2ade708c111132f0f379d"

    _data_check_and_download(data_dir, zip_file, expected_md5sum, overwrite, data_url, named_data_dir)
    data = read_xenium(data_dir)
    logger.info("For this dataset no additional images are available.")
    return data


def xenium_test_dataset(
        overwrite: bool = False,
        output_dir=None,
) -> InSituData:
    """Load the default Xenium test dataset (mouse ileum tiny, v3.0.0, nuclear expansion)."""
    return xenium_test_dataset_v3_nucex(overwrite=overwrite, output_dir=output_dir)


# xenium onboard analysis 3.0.0
# data from https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Mouse_Ileum_tiny/Xenium_Prime_Mouse_Ileum_tiny_outs.zip
# Mouse ileum, nuclear expansion
def xenium_test_dataset_v3_nucex(
        overwrite: bool = False,
        output_dir = None
) -> InSituData:
    """Load mouse ileum tiny Xenium Prime test data (v3.0.0, nuclear expansion segmentation, 10x Genomics)."""
    data_url = "https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Mouse_Ileum_tiny/Xenium_Prime_Mouse_Ileum_tiny_outs.zip"
    base_dir = _resolve_output_dir(output_dir)
    named_data_dir = base_dir / "xenium_test_dataset_v3_nucex"
    data_dir = named_data_dir / "output-XETG00000__slide_id__xenium_test_dataset_v3_nucex"
    zip_file = named_data_dir / Path(data_url).name
    expected_md5sum = "0a7469a005576f2932e4f804dd9bc563"

    _data_check_and_download(data_dir, zip_file, expected_md5sum, overwrite, data_url, named_data_dir)
    data = read_xenium(data_dir)
    logger.info("For this dataset no additional images are available.")
    return data


# xenium onboard analysis 4.0.0
# data from https://cf.10xgenomics.com/samples/xenium/4.0.0/Xenium_V1_Human_Ovary_tiny/Xenium_V1_Human_Ovary_tiny_outs.zip
# Human ovary, nuclear expansion
def xenium_test_dataset_v4_nucex(
        overwrite: bool = False,
        output_dir = None
) -> InSituData:
    """Load human ovary tiny Xenium test data (v4.0.0, nuclear expansion segmentation, 10x Genomics)."""
    data_url = "https://cf.10xgenomics.com/samples/xenium/4.0.0/Xenium_V1_Human_Ovary_tiny/Xenium_V1_Human_Ovary_tiny_outs.zip"
    base_dir = _resolve_output_dir(output_dir)
    named_data_dir = base_dir / "xenium_test_dataset_v4_nucex"
    data_dir = named_data_dir / "output-XETG00000__slide_id__xenium_test_dataset_v4_nucex"
    zip_file = named_data_dir / Path(data_url).name
    expected_md5sum = "a1de61c57b468450ba1fbcdcc1d7c811"

    _data_check_and_download(data_dir, zip_file, expected_md5sum, overwrite, data_url, named_data_dir)
    data = read_xenium(data_dir)
    logger.info("For this dataset no additional images are available.")
    return data

# xenium onboard analysis 4.0.0
# data from https://cf.10xgenomics.com/samples/xenium/4.0.0/Xenium_V1_Protein_Human_Kidney_tiny/Xenium_V1_Protein_Human_Kidney_tiny_outs.zip
# Human kidney, Xenium In Situ Gene and Protein Expression with Cell Segmentation Staining
def xenium_test_dataset_v4_mm(
        overwrite: bool = False,
        output_dir = None
) -> InSituData:
    """Load human ovary tiny Xenium test data (v4.0.0, multimodal cell segmentation, 10x Genomics)."""
    data_url = "https://cf.10xgenomics.com/samples/xenium/4.0.0/Xenium_V1_MultiCellSeg_Human_Ovary_tiny/Xenium_V1_MultiCellSeg_Human_Ovary_tiny_outs.zip"
    base_dir = _resolve_output_dir(output_dir)
    named_data_dir = base_dir / "xenium_test_dataset_v4_mm"
    data_dir = named_data_dir / "output-XETG00000__slide_id__xenium_test_dataset_v4_protein"
    zip_file = named_data_dir / Path(data_url).name
    expected_md5sum = "c0af4c72bed2c7fb4eb6d9f1fdf3b2e1"

    _data_check_and_download(data_dir, zip_file, expected_md5sum, overwrite, data_url, named_data_dir)
    data = read_xenium(data_dir)
    logger.info("For this dataset no additional images are available.")
    return data


# xenium onboard analysis 4.0.0
# data from https://cf.10xgenomics.com/samples/xenium/4.0.0/Xenium_V1_Protein_Human_Kidney_tiny/Xenium_V1_Protein_Human_Kidney_tiny_outs.zip
# Human kidney, Xenium In Situ Gene and Protein Expression with Cell Segmentation Staining
def xenium_test_dataset_v4_protein(
        overwrite: bool = False,
        output_dir = None
) -> InSituData:
    """Load human kidney tiny Xenium test data (v4.0.0, gene and protein expression with cell segmentation staining, 10x Genomics)."""
    data_url = "https://cf.10xgenomics.com/samples/xenium/4.0.0/Xenium_V1_Protein_Human_Kidney_tiny/Xenium_V1_Protein_Human_Kidney_tiny_outs.zip"
    base_dir = _resolve_output_dir(output_dir)
    named_data_dir = base_dir / "xenium_test_dataset_v4_protein"
    data_dir = named_data_dir / "output-XETG00000__slide_id__xenium_test_dataset_v4_protein"
    zip_file = named_data_dir / Path(data_url).name
    expected_md5sum = "50c04dea5e751e1c7508ff24528242e8"

    _data_check_and_download(data_dir, zip_file, expected_md5sum, overwrite, data_url, named_data_dir)
    data = read_xenium(data_dir)
    logger.info("For this dataset no additional images are available.")
    return data


# visium analysis version 2.0.0
# data from https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast
def visium_human_breast_cancer(
        overwrite: bool = False,
        download_fullres: bool = False,
        output_dir = None
) -> InSituData:
    """Load human FFPE breast cancer CytAssist Visium data (v2.0.0, ~3000 spots, 10x Genomics)."""

    # URLs for download
    h5_url = "https://cf.10xgenomics.com/samples/spatial-exp/2.0.0/CytAssist_FFPE_Human_Breast_Cancer/CytAssist_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5"
    spatial_url = "https://cf.10xgenomics.com/samples/spatial-exp/2.0.0/CytAssist_FFPE_Human_Breast_Cancer/CytAssist_FFPE_Human_Breast_Cancer_spatial.tar.gz"
    fullres_url = "https://cf.10xgenomics.com/samples/spatial-exp/2.0.0/CytAssist_FFPE_Human_Breast_Cancer/CytAssist_FFPE_Human_Breast_Cancer_tissue_image.tif"

    # set up paths
    base_dir = _resolve_output_dir(output_dir)
    named_data_dir = base_dir / "visium_hbreastcancer"
    data_dir = named_data_dir / "CytAssist_FFPE_Human_Breast_Cancer"
    h5_file = data_dir / "filtered_feature_bc_matrix.h5"
    spatial_tar = named_data_dir / "CytAssist_FFPE_Human_Breast_Cancer_spatial.tar.gz"
    spatial_dir = data_dir / "spatial"
    fullres_file = data_dir / "CytAssist_FFPE_Human_Breast_Cancer_tissue_image.tif"

    # expected md5sums
    expected_h5_md5sum = "1e3557642aa5a7a5efe9fd07b841109f"
    expected_spatial_md5sum = "26febcbd3d43122e5371a6df2f35e073"
    expected_fullres_md5sum = "9deffd74f614116128c12f3de5be00bc"

    # create data directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)

    # check and download h5 file
    download_h5 = False
    if h5_file.exists():
        logger.info("H5 file exists. Checking md5sum...")
        if _md5sum(h5_file) == expected_h5_md5sum:
            if not overwrite:
                logger.info("The h5 file md5sum matches. Download is skipped. To force download set `overwrite=True`.")
            else:
                logger.info("The h5 file exists but is overwritten because of `overwrite=True`.")
                download_h5 = True
        else:
            logger.warning("The h5 file md5sum doesn't match. File is downloaded.")
            download_h5 = True
    else:
        download_h5 = True

    if download_h5:
        download_url(h5_url, out_dir=data_dir, file_name="filtered_feature_bc_matrix", overwrite=True)

    # check and download spatial data
    download_spatial = False
    if spatial_dir.exists():
        if not overwrite:
            logger.info("Spatial directory exists. Download is skipped. To force download set `overwrite=True`.")
        else:
            logger.info("Spatial directory exists but is overwritten because of `overwrite=True`.")
            download_spatial = True
    else:
        if spatial_tar.exists():
            logger.info("Spatial tar file exists. Checking md5sum...")
            if _md5sum(spatial_tar) == expected_spatial_md5sum:
                if not overwrite:
                    logger.info("The spatial tar md5sum matches. Unpacking...")
                    # unpack and move spatial folder
                    shutil.unpack_archive(spatial_tar, named_data_dir)
                    # move spatial folder to data_dir
                    shutil.move(named_data_dir / "spatial", spatial_dir)
                    # remove tar file
                    os.remove(spatial_tar)
                else:
                    download_spatial = True
            else:
                logger.warning("The spatial tar md5sum doesn't match. File is downloaded.")
                download_spatial = True
        else:
            download_spatial = True

    if download_spatial:
        download_url(spatial_url, out_dir=named_data_dir, overwrite=True)
        # unpack spatial data
        shutil.unpack_archive(spatial_tar, named_data_dir)
        # move spatial folder to data_dir
        if (named_data_dir / "spatial").exists():
            if spatial_dir.exists():
                shutil.rmtree(spatial_dir)
            shutil.move(named_data_dir / "spatial", spatial_dir)
        # remove tar file
        os.remove(spatial_tar)

    # check and download full-resolution image
    if download_fullres:
        download_fullres_img = False
        if fullres_file.exists():
            logger.info("Full-resolution image exists. Checking md5sum...")
            if _md5sum(fullres_file) == expected_fullres_md5sum:
                if not overwrite:
                    logger.info("The full-resolution image md5sum matches. Download is skipped. To force download set `overwrite=True`.")
                else:
                    logger.info("The full-resolution image exists but is overwritten because of `overwrite=True`.")
                    download_fullres_img = True
            else:
                logger.warning("The full-resolution image md5sum doesn't match. File is downloaded.")
                download_fullres_img = True
        else:
            download_fullres_img = True

        if download_fullres_img:
            download_url(fullres_url, out_dir=data_dir, file_name="CytAssist_FFPE_Human_Breast_Cancer_tissue_image", overwrite=True)

    logger.info(f"Visium data structure is ready at {data_dir}")
    logger.info("Dataset contains:")
    logger.info("- filtered_feature_bc_matrix.h5")
    logger.info("- spatial/ directory")
    if download_fullres:
        logger.info("- CytAssist_FFPE_Human_Breast_Cancer_tissue_image.tif")

    # load data into InSituData object
    # Note: You'll need to implement read_visium if not already available
    data = read_visium(data_dir, fullres_pixel_size=0.5476)

    return data
