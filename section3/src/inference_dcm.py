"""
Here we do inference on a DICOM volume, constructing the volume first, and then sending it to the
clinical archive

This code will do the following:
    1. Identify the series to run HippoCrop.AI algorithm on from a folder containing multiple studies
    2. Construct a NumPy volume from a set of DICOM files
    3. Run inference on the constructed volume
    4. Create report from the inference
    5. Call a shell script to push report to the storage archive
"""

import os
import sys
import datetime
import time
import shutil
import subprocess

import numpy as np
import pydicom

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from inference.UNetInferenceAgent import UNetInferenceAgent

def load_dicom_volume_as_numpy_from_list(dcmlist):
    """Loads a list of PyDicom objects a Numpy array.
    Assumes that only one series is in the array

    Arguments:
        dcmlist {list of PyDicom objects} -- path to directory

    Returns:
        tuple of (3D volume, header of the 1st image)
    """

    # In the real world you would do a lot of validation here
    slices = [np.flip(dcm.pixel_array).T for dcm in sorted(dcmlist, key=lambda dcm: dcm.InstanceNumber)]

    # Make sure that you have correctly constructed the volume from your axial slices!
    hdr = dcmlist[0]

    # We return header so that we can inspect metadata properly.
    # Since for our purposes we are interested in "Series" header, we grab header of the
    # first file (assuming that any instance-specific values will be ighored - common approach)
    # We also zero-out Pixel Data since the users of this function are only interested in metadata
    hdr.PixelData = None
    return (np.stack(slices, 2), hdr)

def get_predicted_volumes(pred):
    """Gets volumes of two hippocampal structures from the predicted array

    Arguments:
        pred {Numpy array} -- array with labels. Assuming 0 is bg, 1 is anterior, 2 is posterior

    Returns:
        A dictionary with respective volumes
    """

    # Compute the volume of your hippocampal prediction
    volume_ant = np.sum(pred == 1)
    volume_post = np.sum(pred == 2)
    total_volume = volume_ant + volume_post

    return {"anterior": volume_ant, "posterior": volume_post, "total": total_volume}

def create_report(inference, header, orig_vol, pred_vol):
    """Generates an image with inference report

    Arguments:
        inference {Dictionary} -- dict containing anterior, posterior and full volume values
        header {PyDicom Dataset} -- DICOM header
        orig_vol {Numpy array} -- original volume
        pred_vol {Numpy array} -- predicted label

    Returns:
        PIL image
    """

    # The code below uses PIL image library to compose an RGB image that will go into the report
    # A standard way of storing measurement data in DICOM archives is creating such report and
    # sending them on as Secondary Capture IODs (http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_A.8.html)
    # Essentially, our report is just a standard RGB image, with some metadata, packed into 
    # DICOM format. 

    pimg = Image.new("RGB", (1000, 1000))
    draw = ImageDraw.Draw(pimg)

    header_font = ImageFont.truetype("assets/Roboto-Regular.ttf", size=40)
    main_font = ImageFont.truetype("assets/Roboto-Regular.ttf", size=20)

    #slice_nums = [orig_vol.shape[2]//3, orig_vol.shape[2]//2, orig_vol.shape[2]*3//4] # is there a better choice?
    # It is better tp use the slices with maximum label area
    pred_hip = pred_vol > 0
    slice_1 = pred_hip.sum(axis=tuple((1, 2))).argmax() 
    slice_2 = pred_hip.sum(axis=tuple((0, 2))).argmax() 
    slice_3 = pred_hip.sum(axis=tuple((0, 1))).argmax() 
    slice_nums = [slice_1, slice_2, slice_3]

    # Create the report here and show information that you think would be relevant to
    # clinicians. A sample code is provided below, but feel free to use your creative 
    # genius to make if shine. After all, the is the only part of all our machine learning 
    # efforts that will be visible to the world. The usefulness of your computations will largely
    # depend on how you present them.

    # CUSTOMIZED info for the report   
    draw.text((10, 0), "HippoVolume.AI", (255, 255, 255), font=header_font)
    draw.multiline_text((10, 90), f"Patient ID: {header.PatientID} \
                        \nTotal hippocampal voume:{inference['total']} \
                        \nAnterior voume:{inference['anterior']} \
                        \nPosterior voume:{inference['posterior']}", \
                        (255, 255, 255), font=main_font) 

    # STAND-OUT SUGGESTION:
    # In addition to text data in the snippet above, can you show some images?
    # Think, what would be relevant to show? Can you show an overlay of mask on top of original data?
    # Hint: here's one way to convert a numpy array into a PIL image and draw it inside our pimg object:
    #
    # Create a PIL image from array:
    # Numpy array needs to flipped, transposed and normalized to a matrix of values in the range of [0..255]
    nd_img_1 = np.flip((orig_vol[slice_nums[0],:,:]/np.max(orig_vol))*0xff).T.astype(np.uint8)
    # Ccreate a PIL image from numpy array
    shape_nd_1 = np.flip(6*np.array(nd_img_1.shape), 0)
    pil_i_1 = Image.fromarray(nd_img_1, mode="L").convert("RGBA").resize(shape_nd_1)
    # Paste the PIL image into our main report image object (pimg)
    draw.text((80, 240), f"Sagittal slice #{slice_nums[0]}", (255, 255, 255), font=main_font)
    pimg.paste(pil_i_1, box=(50, 280))
    
    pred_img_1 = np.flip((pred_vol[slice_nums[0],:,:]/np.max(pred_vol))*0xff).T.astype(np.uint8)
    # Ccreate a PIL image from numpy array
    pil_i_1_p = Image.fromarray(pred_img_1, mode="L").convert("RGBA").resize(shape_nd_1)
    #create a mask using RGBA to define an alpha channel to make the overlay transparent
    mask_1 = Image.new('RGBA',pil_i_1_p.size,(0,0,0,100))
    pil_i_1_p_f = Image.composite(pil_i_1, pil_i_1_p,mask_1)
    # Paste the PIL image into our main report image object (pimg)
    draw.text((80, 610), f"Sagittal Mask", (255, 255, 255), font=main_font)
    pimg.paste(pil_i_1_p_f, box=(50, 650))
        
    # second view
    nd_img_2 = np.flip((orig_vol[:, slice_nums[1], :]/np.max(orig_vol))*0xff).T.astype(np.uint8)
    shape_nd_2 = np.flip(6*np.array(nd_img_2.shape), 0)
    pil_i_2 = Image.fromarray(nd_img_2, mode="L").convert("RGBA").resize(shape_nd_2)
    draw.text((130+shape_nd_1[0], 240), f"Coronal slice #{slice_nums[1]}", (255, 255, 255), font=main_font)
    pimg.paste(pil_i_2, box=(100+shape_nd_1[0], 280))
    
    pred_img_2 = np.flip((pred_vol[:, slice_nums[1], :]/np.max(pred_vol))*0xff).T.astype(np.uint8)
    # Ccreate a PIL image from numpy array
    pil_i_2_p = Image.fromarray(pred_img_2, mode="L").convert("RGBA").resize(shape_nd_2)
    #create a mask using RGBA to define an alpha channel to make the overlay transparent
    mask_2 = Image.new('RGBA',pil_i_2_p.size,(0,0,0,100))
    pil_i_2_p_f = Image.composite(pil_i_2, pil_i_2_p,mask_2)
    # Paste the PIL image into our main report image object (pimg)
    draw.text((130+shape_nd_1[0], 610), f"Coronal Mask", (255, 255, 255), font=main_font)
    pimg.paste(pil_i_2_p_f, box=(100+shape_nd_1[0], 650))
    
    # third view
    nd_img_3 = np.flip((orig_vol[:, :, slice_nums[2]]/np.max(orig_vol))*0xff).T.astype(np.uint8)
    shape_nd_3 = np.flip(6*np.array(nd_img_3.shape), 0)
    pil_i_3 = Image.fromarray(nd_img_3, mode="L").convert("RGBA").resize(shape_nd_3)
    draw.text((180+shape_nd_1[0]+shape_nd_2[0], 240), f"Axial slice #{slice_nums[2]}", (255, 255, 255), font=main_font)
    pimg.paste(pil_i_3, box=(150+shape_nd_1[0]+shape_nd_2[0], 280))
    
    pred_img_3 = np.flip((pred_vol[:, :, slice_nums[2]]/np.max(pred_vol))*0xff).T.astype(np.uint8)
    # Ccreate a PIL image from numpy array
    pil_i_3_p = Image.fromarray(pred_img_3, mode="L").convert("RGBA").resize(shape_nd_3)
    #create a mask using RGBA to define an alpha channel to make the overlay transparent
    mask_3 = Image.new('RGBA',pil_i_3_p.size,(0,0,0,100))
    pil_i_3_p_f = Image.composite(pil_i_3, pil_i_3_p,mask_3)
    # Paste the PIL image into our main report image object (pimg)
    draw.text((180+shape_nd_1[0]+shape_nd_2[0], 610), f"Axial Mask", (255, 255, 255), font=main_font)
    pimg.paste(pil_i_3_p_f, box=(150+shape_nd_1[0]+shape_nd_2[0], 650))    

    return pimg

def save_report_as_dcm(header, report, path):
    """Writes the supplied image as a DICOM Secondary Capture file

    Arguments:
        header {PyDicom Dataset} -- original DICOM file header
        report {PIL image} -- image representing the report
        path {Where to save the report}

    Returns:
        N/A
    """

    # Code below creates a DICOM Secondary Capture instance that will be correctly
    # interpreted by most imaging viewers including our OHIF
    # The code here is complete as it is unlikely that as a data scientist you will 
    # have to dive that deep into generating DICOMs. However, if you still want to understand
    # the subject, there are some suggestions below

    # Set up DICOM metadata fields. Most of them will be the same as original file header
    out = pydicom.Dataset(header)

    out.file_meta = pydicom.Dataset()
    out.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    # STAND OUT SUGGESTION: 
    # If you want to understand better the generation of valid DICOM, remove everything below
    # and try writing your own DICOM generation code from scratch.
    # Refer to this part of the standard to see what are the requirements for the valid
    # Secondary Capture IOD: http://dicom.nema.org/medical/dicom/2019e/output/html/part03.html#sect_A.8
    # The Modules table (A.8-1) contains a list of modules with a notice which ones are mandatory (M)
    # and which ones are conditional (C) and which ones are user-optional (U)
    # Note that we are building an RGB image which would have three 8-bit samples per pixel
    # Also note that writing code that generates valid DICOM has a very calming effect
    # on mind and body :)

    out.is_little_endian = True
    out.is_implicit_VR = False

    # We need to change class to Secondary Capture
    out.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    out.file_meta.MediaStorageSOPClassUID = out.SOPClassUID

    # Our report is a separate image series of one image
    out.SeriesInstanceUID = pydicom.uid.generate_uid()
    out.SOPInstanceUID = pydicom.uid.generate_uid()
    out.file_meta.MediaStorageSOPInstanceUID = out.SOPInstanceUID
    out.Modality = "OT" # Other
    out.SeriesDescription = "HippoVolume.AI"

    out.Rows = report.height
    out.Columns = report.width

    out.ImageType = r"DERIVED\PRIMARY\AXIAL" # We are deriving this image from patient data
    out.SamplesPerPixel = 3 # we are building an RGB image.
    out.PhotometricInterpretation = "RGB"
    out.PlanarConfiguration = 0 # means that bytes encode pixels as R1G1B1R2G2B2... as opposed to R1R2R3...G1G2G3...
    out.BitsAllocated = 8 # we are using 8 bits/pixel
    out.BitsStored = 8
    out.HighBit = 7
    out.PixelRepresentation = 0

    # Set time and date
    dt = datetime.date.today().strftime("%Y%m%d")
    tm = datetime.datetime.now().strftime("%H%M%S")
    out.StudyDate = dt
    out.StudyTime = tm
    out.SeriesDate = dt
    out.SeriesTime = tm

    out.ImagesInAcquisition = 1

    # We empty these since most viewers will then default to auto W/L
    out.WindowCenter = ""
    out.WindowWidth = ""

    # Data imprinted directly into image pixels is called "burned in annotation"
    out.BurnedInAnnotation = "YES"

    out.PixelData = report.tobytes()

    pydicom.filewriter.dcmwrite(path, out, write_like_original=False)

def get_series_for_inference(path):
    """Reads multiple series from one folder and picks the one
    to run inference on.

    Arguments:
        path {string} -- location of the DICOM files

    Returns:
        Numpy array representing the series
    """

    # Here we are assuming that path is a directory that contains a full study as a collection
    # of files
    # We are reading all files into a list of PyDicom objects so that we can filter them later
    #dicoms = [pydicom.dcmread(os.path.join(path, f)) for f in os.listdir(path)]

    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(path):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    dicoms = [pydicom.dcmread(f) for f in listOfFiles]
    
    # Create a series_for_inference variable that will contain a list of only 
    # those PyDicom objects that represent files that belong to the series that you 
    # will run inference on.
    # It is important to note that radiological modalities most often operate in terms
    # of studies, and it will most likely be on you to establish criteria for figuring 
    # out which one of the multiple series sent by the scanner is the one you need to feed to 
    # your algorithm. In our case it's rather easy - we have reached an agreement with 
    # people who configured the HippoCrop tool and they label the output of their tool in a 
    # certain way. Can you figure out which is that? 
    # Hint: inspect the metadata of HippoCrop series
    
    series_for_inference = [slice_tmp for slice_tmp in dicoms if slice_tmp.SeriesDescription == 'HippoCrop']
    series_for_inference = sorted(series_for_inference, key = lambda dcm: dcm.ImagePositionPatient[0])

    # Check if there are more than one series (using set comprehension).
    if len({f.SeriesInstanceUID for f in series_for_inference}) != 1:
        print("Error: can not figure out what series to run inference on")
        return []

    return series_for_inference

def os_command(command):
    # Comment this if running under Windows
    sp = subprocess.Popen(["/bin/bash", "-i", "-c", command])
    sp.communicate()

    # Uncomment this if running under Windows
    # os.system(command)

if __name__ == "__main__":
    # This code expects a single command line argument with link to the directory containing
    # routed studies
    if len(sys.argv) != 2:
        print("You should supply one command line argument pointing to the routing folder. Exiting.")
        sys.exit()
    
    # Find all subdirectories within the supplied directory. We assume that 
    # one subdirectory contains a full study
    subdirs = [os.path.join(sys.argv[1], d) for d in os.listdir(sys.argv[1]) if
                os.path.isdir(os.path.join(sys.argv[1], d))]
    
    # Get the latest directory
    study_dir = sorted(subdirs, key=lambda dir: os.stat(dir).st_mtime, reverse=True)[0]

    print(f"Looking for series to run inference on in directory {study_dir}...")
    
    # complete get_series_for_inference
    volume, header = load_dicom_volume_as_numpy_from_list(get_series_for_inference(study_dir))
    print(f"Found series of {volume.shape[2]} axial slices")

    print("HippoVolume.AI: Running inference...")
    # Use the UNetInferenceAgent class and model parameter file from the previous section
    inference_agent = UNetInferenceAgent(
        device="cpu",
        parameter_file_path=r"/home/workspace/src/model.pth")

    # Run inference
    # Single_volume_inference_unpadded takes a volume of arbitrary size 
    # and reshapes y and z dimensions to the patch size used by the model before 
    # running inference. 
    pred_label = inference_agent.single_volume_inference_unpadded(np.array(volume))
    # complete get_predicted_volumes 
    pred_volumes = get_predicted_volumes(pred_label)

    # Create and save the report
    print("Creating and pushing report...")
    report_save_path = r"/home/workspace/out/report.dcm"
    # complete create_report 
    # STAND OUT SUGGESTION: save_report_as_dcm has some suggestions if you want to expand your
    # knowledge of DICOM format
    report_img = create_report(pred_volumes, header, volume, pred_label)
    save_report_as_dcm(header, report_img, report_save_path)

    # Send report to our storage archive
    # TASK: Write a command line string that will issue a DICOM C-STORE request to send our report
    # to our Orthanc server (that runs on port 4242 of the local machine), using storescu tool
    os_command(f'storescu 127.0.0.1 4242 -v -aec TESTSCU +r +sd "{report_save_path}"')

    # This line will remove the study dir if run as root user
    # Sleep to let our StoreSCP server process the report (remember - in our setup
    # the main archive is routing everyting that is sent to it, including our freshly generated
    # report) - we want to give it time to save before cleaning it up
    time.sleep(2)
    shutil.rmtree(study_dir, onerror=lambda f, p, e: print(f"Error deleting: {e[1]}"))

    print(f"Inference successful on {header['SOPInstanceUID'].value}, out: {pred_label.shape}",
          f"volume ant: {pred_volumes['anterior']}, ",
          f"volume post: {pred_volumes['posterior']}, total volume: {pred_volumes['total']}")
