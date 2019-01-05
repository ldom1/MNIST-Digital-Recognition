"""A library containing functions to download, extract and save the MNIST dataset."""

import gzip
import numpy
import os
import six.moves.urllib

def download_extract_save_mnist(source_url, path_to_directory_storage, nb_validation):
    """Downloads the MNIST dataset, extracts it and saves the result of the extraction.

    Parameters
    ----------
    source_url : str
        URL of the MNIST website.
    path_to_directory_storage : str
        Path to the directory storing the downloaded
        MNIST dataset and the result of the extraction.
    nb_validation : int
        Number of images and labels dedicated to
        validation.

    """
    # The 1st string in `filenames` is the name of
    # the file containing the training images and
    # the validation images. The 3rd string in `filename`
    # is the name of the file containing the test images.
    filenames = (
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    )
    for filename in filenames:
        download_if_not_exist(source_url,
                              filename,
                              os.path.join(path_to_directory_storage, filename))

    path_to_training_images = os.path.join(path_to_directory_storage,
                                           'training_images.npy')
    path_to_validation_images = os.path.join(path_to_directory_storage,
                                             'validation_images.npy')
    if os.path.isfile(path_to_training_images) and os.path.isfile(path_to_validation_images):
        print('The files at respectily "{0}" and "{1}" already exist.'.format(path_to_training_images, path_to_validation_images))
    else:
        
        # `training_validation_images_uint8` contains the training
        # images and the validation images.
        training_validation_images_uint8 = extract_images(os.path.join(path_to_directory_storage, filenames[0]))
        
        # The first `nb_validation` images in `training_validation_images_uint8`
        # are set aside for validation.
        numpy.save(path_to_validation_images,
                   training_validation_images_uint8[0:nb_validation, :, :, :])
        
        # The other images are dedicated to training.
        numpy.save(path_to_training_images,
                   training_validation_images_uint8[nb_validation:, :, :, :])

    path_to_training_labels = os.path.join(path_to_directory_storage,
                                           'training_labels.npy')
    path_to_validation_labels = os.path.join(path_to_directory_storage,
                                             'validation_labels.npy')
    if os.path.isfile(path_to_training_labels) and os.path.isfile(path_to_validation_labels):
        print('The files at respectily "{0}" and "{1}" already exist.'.format(path_to_training_labels, path_to_validation_labels))
    else:
        training_validation_labels_uint8 = extract_labels(os.path.join(path_to_directory_storage, filenames[1]))
        
        # The first `nb_validation` labels in `training_validation_labels_uint8`
        # are set aside for validation.
        numpy.save(path_to_validation_labels,
                   training_validation_labels_uint8[0:nb_validation])
        
        # The other labels are dedicated to training.
        numpy.save(path_to_training_labels,
                   training_validation_labels_uint8[nb_validation:])

    path_to_test_images = os.path.join(path_to_directory_storage,
                                       'test_images.npy')
    if os.path.isfile(path_to_test_images):
        print('The file at "{}" already exists.'.format(path_to_test_images))
    else:
        
        # `test_images_uint8` contains the test images.
        test_images_uint8 = extract_images(os.path.join(path_to_directory_storage, filenames[2]))
        numpy.save(path_to_test_images,
                   test_images_uint8)

    path_to_test_labels = os.path.join(path_to_directory_storage,
                                       'test_labels.npy')
    if os.path.isfile(path_to_test_labels):
        print('The file at "{}" already exists.'.format(path_to_test_labels))
    else:
        
        # `test_labels_uint8` contains the test labels.
        test_labels_uint8 = extract_labels(os.path.join(path_to_directory_storage, filenames[3]))
        numpy.save(path_to_test_labels,
                   test_labels_uint8)

def download_if_not_exist(source_url, filename, path_to_downloaded_file):
    """Downloads a file at a given source URL if this file is not already saved.

    Parameters
    ----------
    source_url : str
        URL of the directory containing the file
        to be downloaded.
    filename : str
        Name of the file to be downloaded.
    path_to_downloaded_file : str
        Path to the downloaded file.


    """
    if os.path.isfile(path_to_downloaded_file):
        print('The file at "{}" already exists.'.format(path_to_downloaded_file))
    else:
        six.moves.urllib.request.urlretrieve(os.path.join(source_url, filename),
                                             path_to_downloaded_file)
        print('Dowloaded file saved at "{}".'.format(path_to_downloaded_file))

def extract_images(path_to_downloaded_file):
    """Extracts all the images from the downloaded file ".gz".

    Parameters
    ----------
    path_to_downloaded_file : str
        Path to the downloaded file ".gz".

    Returns
    -------
    numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Extracted images. The 4th array dimension
        is equal to 1.

    Raises
    ------
    ValueError
        If the magic number is not equal to 2051.

    """
    # All the 32-bit integers in the files ".gz" are stored
    # in the MSB first format. It is high endian.
    dtype_int32_high_endian = numpy.dtype(numpy.int32).newbyteorder('>')
    with gzip.open(path_to_downloaded_file, 'rb') as file:
        
        # In the file ".gz", the offset between two successive
        # 32-bit integers is equal to 4.
        magic_number = numpy.frombuffer(file.read(4),
                                        dtype=dtype_int32_high_endian).item()
        if magic_number != 2051:
            raise ValueError('The magic number is not equal to 2051. It is equal to {}.'.format(magic_number))
        nb_images = numpy.frombuffer(file.read(4),
                                     dtype=dtype_int32_high_endian).item()
        nb_rows = numpy.frombuffer(file.read(4),
                                   dtype=dtype_int32_high_endian).item()
        nb_columns = numpy.frombuffer(file.read(4),
                                      dtype=dtype_int32_high_endian).item()
        
        # In the file ".gz", the offset between two successive
        # pixels is equal to 1.
        array_1d_uint8 = numpy.frombuffer(file.read(nb_rows*nb_columns*nb_images),
                                          dtype=numpy.uint8)
        
        # The last singleton dimension in `images_uint8`
        # will be necessary when using convolutions.
        images_uint8 = numpy.reshape(array_1d_uint8,
                                     (nb_images, nb_rows, nb_columns, 1))
    return images_uint8

def extract_labels(path_to_downloaded_file):
    """Extracts all the labels from the downloaded file ".gz".

    Parameters
    ----------
    path_to_downloaded_file : str
        Path to the downloaded file ".gz".

    Returns
    -------
    numpy.ndarray
        1D array with data-type `numpy.uint8`.
        Extracted labels in their standard representation.

    Raises
    ------
    ValueError
        If the magic number is not equal to 2049.

    """
    dtype_int32_high_endian = numpy.dtype(numpy.int32).newbyteorder('>')
    with gzip.open(path_to_downloaded_file, 'rb') as file:
        
        # In the file ".gz", the offset between two successive
        # 32-bit integers is equal to 4.
        magic_number = numpy.frombuffer(file.read(4),
                                        dtype=dtype_int32_high_endian).item()
        if magic_number != 2049:
            raise ValueError('The magic number is not equal to 2049. It is equal to {}.'.format(magic_number))
        nb_images = numpy.frombuffer(file.read(4),
                                     dtype=dtype_int32_high_endian).item()
        
        # `labels_uint8[i]` is the label of the
        # ith example.
        labels_uint8 = numpy.frombuffer(file.read(nb_images),
                                        dtype=numpy.uint8)
    return labels_uint8

def preprocess_images(images_uint8):
    """Preprocesses the images by flattening them and rescaling them to [-0.5, 0.5].

    Parameters
    ----------
    images_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Images to be preprocessed. `images_uint8[i, :, :, :]`
        is the ith image to be preprocessed.

    Returns
    -------
    numpy.ndarray
        2D array with data-type `numpy.float32`.
        Preprocessed images.

    Raises
    ------
    TypeError
        If `images_uint8.dtype` is not equal to `numpy.uint8`.

    """
    if images_uint8.dtype != numpy.uint8:
        raise TypeError('`images_uint8.dtype` is not equal to `numpy.uint8`.')
    reshaped_images_uint8 = numpy.reshape(images_uint8,
                                          (images_uint8.shape[0], -1))
    return reshaped_images_uint8.astype(numpy.float32)/255. - 0.5

def preprocess_labels(labels_uint8):
    """Preprocesses the labels by casting them to 32-bit integer.

    Parameters
    ----------
    labels_uint8 : numpy.ndarray
        1D array with data-type `numpy.uint8`.
        Labels in their standard representation.

    Returns
    -------
    numpy.ndarray
        1D array with data-type `numpy.int32`.
        Preprocessed labels in their standard representation.

    """
    if labels_uint8.dtype != numpy.uint8:
        raise TypeError('`labels_uint8.dtype` is not equal to `numpy.uint8`.')
    return labels_uint8.astype(numpy.int32)


