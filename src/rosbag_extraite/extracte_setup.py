import os
import shutil
from pathlib import Path

import rosbag2_py
from rosidl_runtime_py.utilities import get_message

# === CONFIGURATION BAG ET TYPES ===
def configuration(bag_path: str):
    # Supprimer ancien dossier data
    root = Path("../../out")
    if not root.exists() and not root.is_dir():
        os.mkdir(root)

    # Ouvrir le bag
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    # Récupérer le type de message pour chaque topic
    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}
    topic_names = [t.name for t in topic_types]


    return reader, topic_names, type_map