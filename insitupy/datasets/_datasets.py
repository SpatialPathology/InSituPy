import hashlib
import os.path
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal
import glob

from insitupy.datasets.download import download_url
from insitupy._constants import CACHE

# parameters for download functions
DEMODIR = CACHE / 'demo_datasets'

# functions that each download a dataset into  '~/.cache/InSituPy/demo_dataset'
def md5sum(filePath):
    with open(filePath, 'rb') as fh:
        m = hashlib.md5()
        while True:
            data = fh.read(8192)
            if not data:
                break
            m.update(data)
    return m.hexdigest()

# function that checks the md5sum of the images and returns a boolean.
def md5sum_image_check(file_path : Path, expected_md5sum, overwrite):
    download = False
    if file_path.exists():
        print("Image exists. Checking md5sum...")
        if md5sum(file_path) == expected_md5sum:
            if not overwrite:
                print(f"The md5sum matches. Download is skipped. To force download set `overwrite=True`.")
                return
            else:
                # if the md5sum matches but overwrite=True, the image is still downloaded
                download = True
        else:
            print(f"The md5sum doesn't match. Image is downloaded.")
            download = True
    else:
        download = True

    return download

# function that checks data for md5sum, downloads and unpacks the data.
def data_check_and_download(xeniumdata_dir, zip_file, expected_md5sum, overwrite, xeniumdata_url, named_data_dir):
    # check if the unzipped xenium data exists
    download_xeniumdata = False
    if xeniumdata_dir.exists():
        # if it exists, everything is fine and it is assumed that the dataset was downloaded correctly. Overwrite is still checked.
        if not overwrite:
            print(f"This dataset exists already. Download is skipped. To force download set `overwrite=True`.")
            return
        else:
            print(f"This dataset exists already but is overwritten because of `overwrite=True`.")
            download_xeniumdata = True
    else:
        # if unzipped xenium data does not exist, we need to check if a zip file exists, and if yes if its md5sum is correct    
        if zip_file.exists():
            print("ZIP file exists. Checking md5sum...")
            if md5sum(zip_file) == expected_md5sum:
                if not overwrite:
                    print(f"This dataset exists already. Download is skipped. To force download set `overwrite=True`.")
                    return
                else:
                    # if the md5sum matches but overwrite=True, the data is still downloaded
                    download_xeniumdata = True
            else:
                print("The dataset exists already but the md5sum is not as expected. Dataset is downloaded again.")
                download_xeniumdata = True
        else:
            download_xeniumdata = True
            
    if download_xeniumdata:
        # download xenium data as zip file
        download_url(xeniumdata_url, out_dir=named_data_dir, overwrite=True)
        
        # unzip xenium data
        shutil.unpack_archive(zip_file, xeniumdata_dir)
        
        # move files from outs folder into parent directory and remove the outs folder
        if (xeniumdata_dir / "outs/").exists():
            for f in xeniumdata_dir.glob("outs/*"):
                shutil.move(f, xeniumdata_dir)
            os.rmdir(xeniumdata_dir / "outs")
        
        #remove zip file after unpacking
        os.remove(zip_file)

# xenium onboard analysis version 1.0.1
# data from https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast
def human_breast_cancer(
        overwrite: bool = False
) -> None:
    
    # URLs for download
    xeniumdata_url = "https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_outs.zip"
    he_url = "https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.tif"
    if_url = "https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_if_image.tif"
    
    # set up paths
    named_data_dir = DEMODIR / "hbreastcancer"
    xeniumdata_dir = named_data_dir / "output-XETG00000__slide_id__hbreastcancer"
    image_dir = named_data_dir / "unregistered_images"
    zip_file = named_data_dir / Path(xeniumdata_url).name
    
    # check if file exists and has correct md5sum
    expected_md5sum = '7d42a0b232f92a2e51de1f513b1a44fd'
    expected_he_md5sum = 'fc0d0d38b7c039cc0682e51099f8d841'
    expected_if_md5sum = '929839c64ef8331cfd048a614f5f6829'

    # check if data exists (zipped or unzipped), if yes check md5sum
    # if necessary download data
    data_check_and_download(xeniumdata_dir, zip_file, expected_md5sum, overwrite, xeniumdata_url, named_data_dir)
    
    # download image data
    if md5sum_image_check(image_dir/"slide_id__hbreastcancer__HE__histo.ome.tif", expected_he_md5sum, overwrite):
        download_url(he_url, out_dir=image_dir, file_name="slide_id__hbreastcancer__HE__histo", overwrite = True)

    if md5sum_image_check(image_dir/"slide_id__hbreastcancer__CD20_HER2_DAPI__IF.ome.tif", expected_if_md5sum, overwrite):
        download_url(if_url, out_dir=image_dir, file_name="slide_id__hbreastcancer__CD20_HER2_DAPI__IF", overwrite = True)

# xenium onboard analysis version 1.5.0
# data from https://www.10xgenomics.com/resources/datasets/human-kidney-preview-data-xenium-human-multi-tissue-and-cancer-panel-1-standard
def nondiseased_kidney(
        overwrite: bool = False
) -> None:
    
    # URLs for download
    xeniumdata_url = "https://cf.10xgenomics.com/samples/xenium/1.5.0/Xenium_V1_hKidney_nondiseased_section/Xenium_V1_hKidney_nondiseased_section_outs.zip"
    he_url = "https://cf.10xgenomics.com/samples/xenium/1.5.0/Xenium_V1_hKidney_nondiseased_section/Xenium_V1_hKidney_nondiseased_section_he_image.ome.tif"

    # set up paths
    named_data_dir = DEMODIR / "hkidney"
    xeniumdata_dir = named_data_dir / "output-XETG0000__slide_id__hkidney"
    image_dir = named_data_dir / "unregistered_images"
    zip_file = named_data_dir / Path(xeniumdata_url).name

    # check if file exists and has correct md5sum
    expected_md5sum = '194d5e21b40b27fa8c009d4cbdc3272d'
    expected_he_md5sum = 'e457889aea78bef43834e675f0c58d95'

    # check if data exists (zipped or unzipped), if yes check md5sum
    # if necessary download data
    data_check_and_download(xeniumdata_dir, zip_file, expected_md5sum, overwrite, xeniumdata_url, named_data_dir)
    
    # download image data
    if md5sum_image_check(image_dir/"slide_id__hkidney__HE__histo.ome.tif", expected_he_md5sum, overwrite):
        download_url(he_url, out_dir = image_dir, file_name="slide_id__hkidney__HE__histo", overwrite = True)

# xenium onboard analysis version 1.6.0
# data from https://www.10xgenomics.com/datasets/pancreatic-cancer-with-xenium-human-multi-tissue-and-cancer-panel-1-standard 
def pancreatic_cancer(
        overwrite: bool = False

) -> None:
    
    # URLs for download
    xeniumdata_url = "https://cf.10xgenomics.com/samples/xenium/1.6.0/Xenium_V1_hPancreas_Cancer_Add_on_FFPE/Xenium_V1_hPancreas_Cancer_Add_on_FFPE_outs.zip"
    he_url = "https://cf.10xgenomics.com/samples/xenium/1.6.0/Xenium_V1_hPancreas_Cancer_Add_on_FFPE/Xenium_V1_hPancreas_Cancer_Add_on_FFPE_he_image.ome.tif"
    if_url = "https://cf.10xgenomics.com/samples/xenium/1.6.0/Xenium_V1_hPancreas_Cancer_Add_on_FFPE/Xenium_V1_hPancreas_Cancer_Add_on_FFPE_if_image.ome.tif"

    # set up paths
    named_data_dir = DEMODIR / "hpancreas"
    xeniumdata_dir = named_data_dir / "output-XETG00000__slide_id__hpancreas"
    image_dir = named_data_dir / "unregistered_images"
    zip_file = named_data_dir / Path(xeniumdata_url).name

    # check if file exists and has correct md5sum
    expected_md5sum = '7acca4c2a40f09968b72275403c29f93'
    expected_he_md5sum = '4e96596ea13a3d0f6139638b2b90aef4'
    expected_if_md5sum = 'c859a7ab5d29807b4daf1f66cb6f5060'
    

    # check if data exists (zipped or unzipped), if yes check md5sum
    # if necessary download data)
    data_check_and_download(xeniumdata_dir, zip_file, expected_md5sum, overwrite, xeniumdata_url, named_data_dir)
    
    # download image data
    if md5sum_image_check(image_dir/"slide_id__hPancreas__HE__histo.ome.tif", expected_he_md5sum, overwrite):
        download_url(he_url, out_dir = image_dir, file_name = "slide_id__hPancreas__HE__histo", overwrite = True)

    if md5sum_image_check(image_dir/"slide_id__hPancreas__CD20_TROP2_PPY_DAPI__IF.ome.tif", expected_if_md5sum, overwrite):
        download_url(if_url, out_dir = image_dir, file_name = "slide_id__hPancreas__CD20_TROP2_PPY_DAPI__IF", overwrite = True )
        
# xenium onboard analysis version 1.7.0    
# data from https://www.10xgenomics.com/resources/datasets/human-skin-preview-data-xenium-human-skin-gene-expression-panel-add-on-1-standard
def hskin_melanoma(
        overwrite: bool = False

) -> None:
    
    # URLs for download
    xeniumdata_url = "https://cf.10xgenomics.com/samples/xenium/1.7.0/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE_outs.zip"
    he_url = "https://cf.10xgenomics.com/samples/xenium/1.7.0/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE_he_image.ome.tif"

    # set up paths
    named_data_dir = DEMODIR / "hskin"
    xeniumdata_dir = named_data_dir / "output-XETG00000__slide_id__hskin"
    image_dir = named_data_dir / "unregistered_images"
    zip_file = named_data_dir / Path(xeniumdata_url).name

    # check if file exists and has correct md5sum
    expected_md5sum = '29102799a3f1858c7318b705eb1a8584'
    expected_he_md5sum = '169af7630e0124eef61d252183243a06'

    # check if data exists (zipped or unzipped), if yes check md5sum
    # if necessary download data
    data_check_and_download(xeniumdata_dir, zip_file, expected_md5sum, overwrite, xeniumdata_url, named_data_dir)

    # download image data
    if md5sum_image_check(image_dir/"slide_id__hskin__HE__histo.ome.tif", expected_he_md5sum, overwrite):
        download_url(he_url, out_dir = image_dir, file_name="slide_id__hskin__HE__histo", overwrite = True)

# xenium onboard analysis version 2.0.0
# data from https://www.10xgenomics.com/datasets/ffpe-human-brain-cancer-data-with-human-immuno-oncology-profiling-panel-and-custom-add-on-1-standard
def human_brain_cancer(
        overwrite: bool = False

) -> None:
    
    # URLs for download
    xeniumdata_url = "https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/2.0.0/Xenium_V1_Human_Brain_GBM_FFPE/Xenium_V1_Human_Brain_GBM_FFPE_outs.zip"
    he_url = "https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_Human_Brain_GBM_FFPE/Xenium_V1_Human_Brain_GBM_FFPE_he_image.ome.tif"

    # set up paths
    named_data_dir = DEMODIR / "hbraincancer"
    xeniumdata_dir = named_data_dir / "output-XETG00000__slide_id__hbraincancer"
    image_dir = named_data_dir / "unregistered_images"
    zip_file = named_data_dir / Path(xeniumdata_url).name

    # check if file exists and has correct md5sum
    expected_md5sum = "c116017ad9884cf6944c6c4815bffb3c"
    expected_he_md5sum = "22b66c6e7669933e50a9665d467e639f"

    # check if data exists (zipped or unzipped), if yes check md5sum
    # if necessary download data
    data_check_and_download(xeniumdata_dir, zip_file, expected_md5sum, overwrite, xeniumdata_url, named_data_dir)

    # download image data
    if md5sum_image_check(image_dir/"slide_id__hbraincancer__HE__histo.ome.tif", expected_he_md5sum, overwrite):
        download_url(he_url, out_dir = image_dir, file_name="slide_id__hbraincancer__HE__histo", overwrite = True)