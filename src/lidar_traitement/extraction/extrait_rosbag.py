import os
from typing import List, Any

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def detect_cloud_topics(bag_path: str) -> List[str]:
    """
    Détecte les topics contenant des nuages de points dans un fichier ROS bag.

    Args:
        bag_path (str): Chemin vers le fichier ROS bag.

    Returns:
        List[str]: Liste des noms des topics contenant des nuages de points.
    """
    import rosbag2_py

    # Déterminer le type de stockage
    storage_id = "mcap" if any(
        f.endswith(".mcap") for f in os.listdir(bag_path)
    ) else "sqlite3"

    # Ouvrir le lecteur sans filtre pour obtenir tous les topics
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr"),
    )

    # Obtenir tous les topics et leurs types
    all_topics_and_types = reader.get_all_topics_and_types()

    # Filtrer les topics de type PointCloud2
    cloud_topics = []
    for topic_info in all_topics_and_types:
        if topic_info.type == "sensor_msgs/msg/PointCloud2":
            cloud_topics.append(topic_info.name)

    return cloud_topics


def _open_bag_reader(bag_path: str, topics: List[str]):
    """Ouvre un SequentialReader filtré sur les topics demandés."""
    import rosbag2_py
    storage_id = "mcap" if any(
        f.endswith(".mcap") for f in os.listdir(bag_path)
    ) else "sqlite3"

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr"),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=topics))

    topic_types = {m.name: m.type for m in reader.get_all_topics_and_types()}
    return reader, topic_types

def _open_bag_extrait(reader : Any, topic_types : dict[Any, Any], topics : List[str])-> Any:
    deser = {}
    for topic in (topics):
        t = topic_types.get(topic)
        deser[topic] = get_message(t) if t else None

    result = {cle: [] for cle in topics}

    while reader.has_next():
        topic, data, ts = reader.read_next()
        d = deser.get(topic)
        if d is None:
            continue

        msg = deserialize_message(data, d)

        if topic in result:
            result[topic].append((ts, msg))

    return result