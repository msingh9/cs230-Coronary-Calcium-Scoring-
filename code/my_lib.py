import plistlib
import numpy as np
import scipy.ndimage as measurements

# get pixel coordinates
def get_pix_coords(pxy):
    x, y = eval(pxy)
    x = int(x)
    y = int(y)
    assert x > 0 and x < 512, f"Invalid {x} value for pixel coordinate"
    assert y > 0 and y < 512, f"Invalid {y} value for pixel coordinate"
    return (x, y)

# process calcium_xml
# return data is:
#  {<image_index>: [{cid: <integer>, pixels: [(x1,y1), (x2,y2)..]},..]
def process_xml(f):
    # cornary name to id
    cornary_name_2_id = {"Right Coronary Artery": 0,
                         "Left Anterior Descending Artery": 1,
                         "Left Coronary Artery": 2,
                         "Left Circumflex Artery": 3}

    # input XML file
    # output - directory containing various meta data
    with open(f, 'rb') as fin:
        pl = plistlib.load(fin)
    # extract needed info from XML
    data = {}
    for image in pl["Images"]:
        iidx = image["ImageIndex"]
        num_rois = image["NumberOfROIs"]
        assert num_rois == len(image["ROIs"]), f"{num_rois} ROIs but not all specified in {f}"
        for roi in image["ROIs"]:
            if (len(roi['Point_px']) > 0):
                if iidx not in data:
                    data[iidx] = []
                if roi['Name'] not in cornary_name_2_id:
                    print (f"ERROR: Missing name {roi['Name']} in cornary_name_2_id dict for file {f}")
                    # fixme (identify and fix labels)
                    roi['Name'] = "Left Coronary Artery"
                data[iidx].append({"cid" : cornary_name_2_id[roi['Name']]})
                assert len(roi['Point_px']) == roi['NumberOfPoints'], f"Number of ROI points does not match with given length for {f}"
                data[iidx][-1]['pixels'] = [get_pix_coords(pxy) for pxy in roi['Point_px']]
            else:
                if 0:
                    print (f"Warning: ROI without pixels specified for {iidx} in {f}")

    return data


def get_object_agatston(calc_object: np.ndarray, calc_pixel_count: int):
    """Applies standard categorization: https://radiopaedia.org/articles/agatston-score"""
    object_max = np.max(calc_object)
    object_agatston = 0
    if 130 <= object_max < 200:
        object_agatston = calc_pixel_count * 1
    elif 200 <= object_max < 300:
        object_agatston = calc_pixel_count * 2
    elif 300 <= object_max < 400:
        object_agatston = calc_pixel_count * 3
    elif object_max >= 400:
        object_agatston = calc_pixel_count * 4
    # print(f'For {calc_pixel_count} with max {object_max} returning AG of {object_agatston}')
    return object_agatston


def compute_agatston_for_slice(X, Y, min_calc_object_pixels=3) -> int:
    # FIXME: get from dicom image
    dicom_attributes = {}
    dicom_attributes['slice_thickness'] = 5
    dicom_attributes['pixel_spacing'] = (0.404296875, 0.404296875)
    dicom_attributes['rescale_intercept'] = -1024.
    dicom_attributes['rescale_slope'] = 1.0

    def create_hu_image(X):
        norm_const = np.array(2 ** 16 - 1).astype('float32')
        return X * norm_const * dicom_attributes['rescale_slope'] - dicom_attributes['rescale_intercept']

    mask = Y[:, :, 0]
    if np.sum(mask) == 0:
        return 0
    slice_agatston = 0
    pixel_volume = (dicom_attributes['slice_thickness']
                    * dicom_attributes['pixel_spacing'][0]
                    * dicom_attributes['pixel_spacing'][1])/3

    hu_image = create_hu_image(X)
    labeled_mask, num_labels = measurements.label(mask,
                                                  structure=np.ones((3, 3)))
    for calc_idx in range(1, num_labels + 1):
        label = np.zeros(mask.shape)
        label[labeled_mask == calc_idx] = 1
        calc_object = hu_image * label

        calc_pixel_count = np.sum(label)
        # Remove small calcified objects.
        if calc_pixel_count <= min_calc_object_pixels:
            continue
        calc_volume = calc_pixel_count * pixel_volume
        object_agatston = round(get_object_agatston(calc_object, calc_volume))
        slice_agatston += object_agatston
    return slice_agatston


