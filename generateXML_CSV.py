#---------------------------------------------------------------------
#
# Creating Darkflow Annotations for dataset (Converting in XML format)
#
# Function = write_xml()
#
# By - Jatin Kumar Mandav
#
# Requires 9 arguments, and 2 default arguments if you wish to alter
#
# Arguments:
#   Directory where to save the XML file
#   Label Name of the Image
#   Name of the Image
#   Boundung Box x1
#   Boundung Box y1
#   Boundung Box x2
#   Boundung Box y2
#   Width of Image
#   Height of Image
#   Depth of Image or Channels value '3'(RGB) by default
#   Pose of Image, "Unspecified" by default
#
#---------------------------------------------------------------------

from lxml import etree
import xml.etree.cElementTree as ET

def write_xml(savedir, label_name, imagename, bboxx1, bboxy1, bboxx2, bboxy2, imgWidth, imgHeight, depth=3, pose="Unspecified"):
    individual_images_read_from_csv = pd.read_csv(csvFile).values

    currentfolder = savedir.split("\\")[-1]

    for image in individual_images_read_from_csv:

        annotation = ET.Element("annotaion")
        ET.SubElement(annotation, 'folder').text = str(currentfolder)
        ET.SubElement(annotation, 'filename').text = str(imagename)
        size = ET.SubElement(annotation, 'size')
        ET.SubElement(size, 'width').text = str(imgWidth)
        ET.SubElement(size, 'height').text = str(imgHeight)
        ET.SubElement(size, 'depth').text = str(depth)

        ET.SubElement(annotation, 'segmented').text = '0'

        obj = ET.SubElement(annotation, 'object')

        ET.SubElement(obj, 'name').text = str(label_name)
        ET.SubElement(obj, 'pose').text = str(pose)
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'

        bbox = ET.SubElement(obj, 'bndbox')

        ET.SubElement(bbox, 'xmin').text = str(bboxx1)
        ET.SubElement(bbox, 'ymin').text = str(bboxy1)
        ET.SubElement(bbox, 'xmax').text = str(bboxx2)
        ET.SubElement(bbox, 'ymax').text = str(bboxy2)

        xml_str = ET.tostring(annotation)
        root = etree.fromstring(xml_str)
        xml_str = etree.tostring(root, pretty_print=True)

        save_path = os.path.join(saveDir, imagename + ".xml")
        with open(save_path, 'wb') as temp_xml:
            temp_xml.write(xml_str)
