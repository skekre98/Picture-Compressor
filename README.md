# Picture-Compressor
A tool for compressing images using unsupervised machine learning


## K-means Clustering

K-means is an unsupervised algorithm that makes inferences from datasets using only input vectors without referring to known, or labelled, outcomes. K-means algorithm identifies k number of centroids, and then allocates every data point to the nearest cluster, while keeping the centroids as small as possible. In reference to this tool, your k centroids are the number of colors that are selected are majority colors. For example, if k were to be 5, then 5 colors would show up in your compressed image. 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Dependencies

* [NumPy](https://www.numpy.org) - package for scientific computing with Python
* [Pillow](https://pillow.readthedocs.io/en/stable/) - Python Imaging Library
* [scikit-learn](https://scikit-learn.org/stable/) - machine learning library for the Python 


