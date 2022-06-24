# Copyright (C) 2013 Oskar Maier
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# author Oskar Maier
# version r0.3.0
# since 2012-01-18
# status Release

# build-in modules
import inspect

# third-party modules
import scipy

# own modules
from ..core import Logger
from .graph import GCGraph
from medpy.graphcut.energy_label import __check_label_image

def graph_from_voxels(fg_markers,
                        bg_markers,
                        regional_term = False,
                        boundary_term = False,
                        regional_term_args = False,
                        boundary_term_args = False):
    """
    Create a graph-cut ready graph to segment a nD image using the voxel neighbourhood.
    
    Create a `~medpy.graphcut.maxflow.GraphDouble` object for all voxels of an image with a
    :math:`ndim * 2` neighbourhood.
    
    Every voxel of the image is regarded as a node. They are connected to their immediate
    neighbours via arcs. If to voxels are neighbours is determined using
    :math:`ndim*2`-connectedness (e.g. :math:`3*2=6` for 3D). In the next step the arcs weights
    (n-weights) are computed using the supplied ``boundary_term`` function
    (see :mod:`~medpy.graphcut.energy_voxel` for a selection).
    
    Implicitly the graph holds two additional nodes: the source and the sink, so called
    terminal nodes. These are connected with all other nodes through arcs of an initial
    weight (t-weight) of zero.
    All voxels that are under the foreground markers are considered to be tightly bound
    to the source: The t-weight of the arc from source to these nodes is set to a maximum
    value. The same goes for the background markers: The covered voxels receive a maximum
    (`~medpy.graphcut.graph.GCGraph.MAX`) t-weight for their arc towards the sink.
    
    All other t-weights are set using the supplied ``regional_term`` function
    (see :mod:`~medpy.graphcut.energy_voxel` for a selection).
    
    Parameters
    ----------
    fg_markers : ndarray
        The foreground markers as binary array of the same shape as the original image.
    bg_markers : ndarray
        The background markers as binary array of the same shape as the original image.
    regional_term : function
        This can be either `False`, in which case all t-weights are set to 0, except for
        the nodes that are directly connected to the source or sink; or a function, in
        which case the supplied function is used to compute the t_edges. It has to
        have the following signature *regional_term(graph, regional_term_args)*, and is
        supposed to compute (source_t_weight, sink_t_weight) for all voxels of the image
        and add these to the passed `~medpy.graphcut.graph.GCGraph` object. The weights
        have only to be computed for nodes where they do not equal zero. Additional
        parameters can be passed to the function via the ``regional_term_args`` parameter.
    boundary_term : function
        This can be either `False`, in which case all n-edges, i.e. between all nodes
        that are not source or sink, are set to 0; or a function, in which case the
        supplied function is used to compute the edge weights. It has to have the
        following signature *boundary_term(graph, boundary_term_args)*, and is supposed
        to compute the edges between the graphs nodes and to add them to the supplied
        `~medpy.graphcut.graph.GCGraph` object. Additional parameters can be passed to
        the function via the ``boundary_term_args`` parameter.
    regional_term_args : tuple
        Use this to pass some additional parameters to the ``regional_term`` function.
    boundary_term_args : tuple    
        Use this to pass some additional parameters to the ``boundary_term`` function.
    
    Returns
    -------
    graph : `~medpy.graphcut.maxflow.GraphDouble`
        The created graph, ready to execute the graph-cut.
    
    Raises
    ------
    AttributeError
        If an argument is malformed.
    FunctionError
        If one of the supplied functions returns unexpected results.
    
    Notes
    -----
    If a voxel is marked as both, foreground and background, the background marker
    is given higher priority.
     
    All arcs whose weight is not explicitly set are assumed to carry a weight of zero.
    """
    # prepare logger
    logger = Logger.getInstance()
    
    # prepare result graph
    logger.debug('Assuming {} nodes and {} edges for image of shape {}'.format(fg_markers.size, __voxel_4conectedness(fg_markers.shape), fg_markers.shape)) 
    graph = GCGraph(fg_markers.size, __voxel_4conectedness(fg_markers.shape))
    
    logger.info('Performing attribute tests...')
    
    # check, set and convert all supplied parameters
    fg_markers = scipy.asarray(fg_markers, dtype=scipy.bool_)
    bg_markers = scipy.asarray(bg_markers, dtype=scipy.bool_)
    
    # set dummy functions if not supplied
    if not regional_term: regional_term = __regional_term_voxel
    if not boundary_term: boundary_term = __boundary_term_voxel
    
    # check supplied functions and their signature
    if not hasattr(regional_term, '__call__') or not 2 == len(inspect.getargspec(regional_term)[0]):
        raise AttributeError('regional_term has to be a callable object which takes two parameter.')
    if not hasattr(boundary_term, '__call__') or not 2 == len(inspect.getargspec(boundary_term)[0]):
        raise AttributeError('boundary_term has to be a callable object which takes two parameters.')

    logger.debug('#nodes={}, #hardwired-nodes source/sink={}/{}'.format(fg_markers.size,
                                                                        len(fg_markers.ravel().nonzero()[0]),
                                                                        len(bg_markers.ravel().nonzero()[0])))
    
    # compute the weights of all edges from the source and to the sink i.e.
    # compute the weights of the t_edges Wt
    logger.info('Computing and adding terminal edge weights...')
    regional_term(graph, regional_term_args)

    # compute the weights of the edges between the neighbouring nodes i.e.
    # compute the weights of the n_edges Wr
    logger.info('Computing and adding inter-node edge weights...')
    boundary_term(graph, boundary_term_args)
    
    # collect all voxels that are under the foreground resp. background markers i.e.
    # collect all nodes that are connected to the source resp. sink
    logger.info('Setting terminal weights for the markers...')
    if not 0 == scipy.count_nonzero(fg_markers):
        graph.set_source_nodes(fg_markers.ravel().nonzero()[0])
    if not 0 == scipy.count_nonzero(bg_markers):
        graph.set_sink_nodes(bg_markers.ravel().nonzero()[0])    
    
    return graph.get_graph()

def graph_from_labels(label_image,
                        fg_markers,
                        bg_markers,
                        regional_term = False,
                        boundary_term = False,
                        regional_term_args = False,
                        boundary_term_args = False):
    """
    Create a graph-cut ready graph to segment a nD image using the region neighbourhood.
    
    Create a `~medpy.graphcut.maxflow.GraphDouble` object for all regions of a nD label
    image.
    
    Every region of the label image is regarded as a node. They are connected to their
    immediate neighbours by arcs. If to regions are neighbours is determined using
    :math:`ndim*2`-connectedness (e.g. :math:`3*2=6` for 3D).
    In the next step the arcs weights (n-weights) are computed using the supplied
    ``boundary_term`` function (see :mod:`~medpy.graphcut.energy_voxel` for a selection).
    
    Implicitly the graph holds two additional nodes: the source and the sink, so called
    terminal nodes. These are connected with all other nodes through arcs of an initial
    weight (t-weight) of zero.
    All regions that are under the foreground markers are considered to be tightly bound
    to the source: The t-weight of the arc from source to these nodes is set to a maximum 
    value. The same goes for the background markers: The covered regions receive a
    maximum (`~medpy.graphcut.graph.GCGraph.MAX`) t-weight for their arc towards the sink.
    
    All other t-weights are set using the supplied ``regional_term`` function
    (see :mod:`~medpy.graphcut.energy_voxel` for a selection).
    
    Parameters
    ----------
    label_image: ndarray
        The label image as an array cwhere each voxel carries the id of the region it
        belongs to. Note that the region labels have to start from 1 and be continuous
        (can be achieved with `~medpy.filter.label.relabel`).
    fg_markers : ndarray
        The foreground markers as binary array of the same shape as the original image.
    bg_markers : ndarray
        The background markers as binary array of the same shape as the original image.
    regional_term : function
        This can be either `False`, in which case all t-weights are set to 0, except for
        the nodes that are directly connected to the source or sink; or a function, in
        which case the supplied function is used to compute the t_edges. It has to
        have the following signature *regional_term(graph, regional_term_args)*, and is
        supposed to compute (source_t_weight, sink_t_weight) for all regions of the image
        and add these to the passed `~medpy.graphcut.graph.GCGraph` object. The weights
        have only to be computed for nodes where they do not equal zero. Additional
        parameters can be passed to the function via the ``regional_term_args`` parameter.
    boundary_term : function
        This can be either `False`, in which case all n-edges, i.e. between all nodes
        that are not source or sink, are set to 0; or a function, in which case the
        supplied function is used to compute the edge weights. It has to have the
        following signature *boundary_term(graph, boundary_term_args)*, and is supposed
        to compute the edges between all adjacent regions of the image and to add them
        to the supplied `~medpy.graphcut.graph.GCGraph` object. Additional parameters
        can be passed to the function via the ``boundary_term_args`` parameter.
    regional_term_args : tuple
        Use this to pass some additional parameters to the ``regional_term`` function.
    boundary_term_args : tuple    
        Use this to pass some additional parameters to the ``boundary_term`` function.

    Returns
    -------
    graph : `~medpy.graphcut.maxflow.GraphDouble`
        The created graph, ready to execute the graph-cut.
    
    Raises
    ------
    AttributeError
        If an argument is malformed.
    FunctionError
        If one of the supplied functions returns unexpected results.
    
    Notes
    -----
    If a voxel is marked as both, foreground and background, the background marker
    is given higher priority.
     
    All arcs whose weight is not explicitly set are assumed to carry a weight of zero.    
    """    
    # prepare logger
    logger = Logger.getInstance()
    
    logger.info('Performing attribute tests...')
    
    # check, set and convert all supplied parameters
    label_image = scipy.asarray(label_image)
    fg_markers = scipy.asarray(fg_markers, dtype=scipy.bool_)
    bg_markers = scipy.asarray(bg_markers, dtype=scipy.bool_)
    
    __check_label_image(label_image)
    
    # set dummy functions if not supplied
    if not regional_term: regional_term = __regional_term_label
    if not boundary_term: boundary_term = __boundary_term_label
    
    # check supplied functions and their signature
    if not hasattr(regional_term, '__call__') or not 3 == len(inspect.getargspec(regional_term)[0]):
        raise AttributeError('regional_term has to be a callable object which takes three parameters.')
    if not hasattr(boundary_term, '__call__') or not 3 == len(inspect.getargspec(boundary_term)[0]):
        raise AttributeError('boundary_term has to be a callable object which takes three parameters.')    
    
    logger.info('Determining number of nodes and edges.')
    
    # compute number of nodes and edges
    nodes = len(scipy.unique(label_image))
    # POSSIBILITY 1: guess the number of edges (in the best situation is faster but requires a little bit more memory. In the worst is slower.)
    edges = 10 * nodes
    logger.debug('guessed: #nodes={} nodes / #edges={}'.format(nodes, edges))
    # POSSIBILITY 2: compute the edges (slow)
    #edges = len(__compute_edges(label_image))
    #logger.debug('computed: #nodes={} nodes / #edges={}'.format(nodes, edges))
        
    # prepare result graph
    graph = GCGraph(nodes, edges)
                                        
    logger.debug('#hardwired-nodes source/sink={}/{}'.format(len(scipy.unique(label_image[fg_markers])),
                                                             len(scipy.unique(label_image[bg_markers]))))
                 
    #logger.info('Extracting the regions bounding boxes...')
    # extract the bounding boxes
    #bounding_boxes = find_objects(label_image)
        
    # compute the weights of all edges from the source and to the sink i.e.
    # compute the weights of the t_edges Wt
    logger.info('Computing and adding terminal edge weights...')
    #regions = set(graph.get_nodes()) - set(graph.get_source_nodes()) - set(graph.get_sink_nodes())
    regional_term(graph, label_image, regional_term_args) # bounding boxes indexed from 0 # old version: regional_term(graph, label_image, regions, bounding_boxes, regional_term_args)

    # compute the weights of the edges between the neighbouring nodes i.e.
    # compute the weights of the n_edges Wr
    logger.info('Computing and adding inter-node edge weights...')
    boundary_term(graph, label_image, boundary_term_args)
    
    # collect all regions that are under the foreground resp. background markers i.e.
    # collect all nodes that are connected to the source resp. sink
    logger.info('Setting terminal weights for the markers...')
    graph.set_source_nodes(scipy.unique(label_image[fg_markers] - 1)) # requires -1 to adapt to node id system
    graph.set_sink_nodes(scipy.unique(label_image[bg_markers] - 1))
    
    return graph.get_graph()

def __regional_term_voxel(graph, regional_term_args):
    """Fake regional_term function with the appropriate signature."""
    return {}

def __regional_term_label(graph, label_image, regional_term_args):
    """Fake regional_term function with the appropriate signature."""
    return {}

def __boundary_term_voxel(graph, boundary_term_args):
    """Fake regional_term function with the appropriate signature."""
    # supplying no boundary term contradicts the whole graph cut idea.
    return {}

def __boundary_term_label(graph, label_image, boundary_term_args):
    """Fake regional_term function with the appropriate signature."""
    # supplying no boundary term contradicts the whole graph cut idea.
    return {}
    
def __voxel_4conectedness(shape):
    """
    Returns the number of edges for the supplied image shape assuming 4-connectedness.
    
    The name of the function has historical reasons. Essentially it returns the number
    of edges assuming 4-connectedness only for 2D. For 3D it assumes 6-connectedness,
    etc.
    
    @param shape the shape of the image
    @type shape sequence
    @return the number of edges
    @rtype int
    """
    shape = list(shape)
    while 1 in shape: shape.remove(1) # empty resp. 1-sized dimensions have to be removed (equal to scipy.squeeze on the array)
    return int(round(sum([(dim - 1)/float(dim) for dim in shape]) * scipy.prod(shape)))
