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
# version r0.1.0
# since 2012-02-06
# status Release

# build-in modules

# third-party modules

# own modules

# code
def graph_to_dimacs(g, f):
    """
    Persists the supplied graph in valid dimacs format into the file.
    
    Parameters
    ----------
    g : `~medpy.graphcut.graph.Graph`
        A graph object to persist.
    f : file
        A file-like object.
    """
    # write comments
    f.write('c Created by medpy\n')
    f.write('c Oskar Maier, oskar.maier@googlemail.com\n')
    f.write('c\n')
    
    # write problem
    f.write('c problem line\n')
    f.write('p max {} {}\n'.format(g.get_node_count() + 2, len(g.get_edges()))) # +2 as terminal nodes also count in dimacs format # no-nodes / no-edges
    
    # denote source and sink
    f.write('c source descriptor\n')
    f.write('n 1 s\n')
    f.write('c sink descriptor\n')
    f.write('n 2 t\n')
    
    # write terminal arcs (t-weights)
    f.write('c terminal arcs (t-weights)\n')
    for node, weight in list(g.get_tweights().items()):
        # Note: the nodes ids of the graph start from 1, but 1 and 2 are reserved for source and sink respectively, therefore add 2
        if not 0 == weight[0]: # 0 weights are implicit
            f.write('a 1 {} {}\n'.format(node + 2, weight[0]))
        if not 0 == weight[1]: # 0 weights are implicit
            f.write('a {} 2 {}\n'.format(node + 2, weight[1]))
    
    # write inter-node arcs (n-weights)
    f.write('c inter-node arcs (n-weights)\n')
    for edge, weight in list(g.get_nweights().items()):
        if not 0 == weight[0]: # 0 weights are implicit
            f.write('a {} {} {}\n'.format(edge[0] + 2, edge[1] + 2, weight[0]))
        # reversed weights have to follow directly in the next line
        if not 0 == weight[1]: # 0 weights are implicit
            f.write('a {} {} {}\n'.format(edge[1] + 2, edge[0] + 2, weight[1]))
            
    # end comment
    f.write('c end-of-file')
