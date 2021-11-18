import plistlib

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

