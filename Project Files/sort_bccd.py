import os
import shutil
import xml.etree.ElementTree as ET

# Paths
ANNOTATIONS_DIR = r'BCCD_Dataset-master\BCCD\Annotations'
IMAGES_DIR = r'BCCD_Dataset-master\BCCD\JPEGImages'
OUTPUT_DIR = r'BCCD_Sorted'

# Use actual BCCD labels
classes = ['WBC', 'RBC', 'Platelets']
counts = {cls: 0 for cls in classes}
skipped = []

# Ensure output folders exist
for cls in classes:
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

# Process XML files
for filename in os.listdir(ANNOTATIONS_DIR):
    if not filename.endswith('.xml'):
        continue

    xml_path = os.path.join(ANNOTATIONS_DIR, filename)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    labels = set()
    for obj in root.findall('object'):
        label = obj.find('name').text
        if label in classes:
            labels.add(label)

    # Skip if multi-label or no valid label
    if len(labels) != 1:
        skipped.append(filename)
        continue

    label = labels.pop()
    image_filename = filename.replace('.xml', '.jpg')
    src_path = os.path.join(IMAGES_DIR, image_filename)
    dst_path = os.path.join(OUTPUT_DIR, label, image_filename)

    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        counts[label] += 1
    else:
        skipped.append(filename)

# ✅ Report
print("✅ Sorting complete.\n")
for cls, count in counts.items():
    print(f"{cls}: {count} images")

print(f"\n❌ Skipped: {len(skipped)} files")
for i, name in enumerate(skipped[:10]):
    print(f"   - {name}")
if len(skipped) > 10:
    print(f"   ...and {len(skipped) - 10} more skipped.")
