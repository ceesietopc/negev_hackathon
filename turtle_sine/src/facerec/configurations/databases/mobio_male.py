#!/usr/bin/env python

import bob.db.mobio
import facereclib

mobio_image_directory = "[YOUR_MOBIO_IMAGE_DIRECTORY]"
mobio_annotation_directory = "[YOUR_MOBIO_ANNOTATION_DIRECTORY]"

database = facereclib.databases.DatabaseBobZT(
    database = bob.db.mobio.Database(
        original_directory = mobio_image_directory,
        original_extension = ".png",
        annotation_directory = mobio_annotation_directory,
    ),
    name = "mobio",
    protocol = 'male',

    all_files_options = { 'gender' : 'male' },
    extractor_training_options = { 'gender' : 'male' },
    projector_training_options = { 'gender' : 'male' },
    enroller_training_options = { 'gender' : 'male' },
    z_probe_options = { 'gender' : 'male' }

)
