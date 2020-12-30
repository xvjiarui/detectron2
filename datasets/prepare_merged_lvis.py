#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import copy
import json
import os


def merge_jsons(input_jsons):
    new_json = copy.deepcopy(input_jsons[0])
    for input_json in input_jsons[1:]:
        new_json["annotations"].extend(input_json["annotations"])
        new_json["images"].extend(input_json["images"])

    return new_json


def merge_lvis(input_filenames, exclude_filenames, output_filename):
    """
    Filter LVIS instance segmentation annotations to remove all categories that are not included in
    COCO. The new json files can be used to evaluate COCO AP using `lvis-api`. The category ids in
    the output json are the incontiguous COCO dataset ids.

    Args:
        input_filename (str): path to the LVIS json file.
        output_filename (str): path to the COCOfied json file.
    """

    input_jsons = []
    for input_filename in input_filenames:
        with open(input_filename, "r") as f:
            input_jsons.append(json.load(f))

    merged_input = merge_jsons(input_jsons)

    exclude_jsons = []
    for exclude_json in exclude_filenames:
        with open(exclude_json, "r") as f:
            exclude_jsons.append(json.load(f))

    merged_exclude = merge_jsons(exclude_jsons)

    exclude_image_ids = set(_["id"] for _ in merged_exclude["images"])
    delete_image_indices = []
    for i, image in enumerate(merged_input["images"]):
        if image["id"] in exclude_image_ids:
            delete_image_indices.append(i)

    delete_annotation_indices = []
    for i, annotation in enumerate(merged_input["annotations"]):
        if annotation["image_id"] in exclude_image_ids:
            delete_annotation_indices.append(i)

    for i in sorted(delete_image_indices, reverse=True):
        merged_input["images"].pop(i)

    for i in sorted(delete_annotation_indices, reverse=True):
        merged_input["annotations"].pop(i)

    with open(output_filename, "w") as f:
        json.dump(merged_input, f)


if __name__ == "__main__":
    dataset_dir = os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), "lvis")
    merge_lvis(
        [
            os.path.join(dataset_dir, "lvis_v1_train.json"),
            os.path.join(dataset_dir, "lvis_v1_val.json"),
        ],
        [os.path.join(dataset_dir, "lvis_v0.5_val.json")],
        os.path.join(dataset_dir, "lvis_v1.5_train.json"),
    )

    with open(os.path.join(dataset_dir, "lvis_v1.5_train.json"), "r") as f:
        merged = json.load(f)

    with open(os.path.join(dataset_dir, "lvis_v0.5_val.json"), "r") as f:
        excluded = json.load(f)

    assert set(_["id"] for _ in merged["images"]).isdisjoint(
        set(_["id"] for _ in excluded["images"])
    )

    print("verified")
