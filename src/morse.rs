use petgraph::graph::{Graph, NodeIndex};
use petgraph::unionfind::UnionFind;

use std::collections::{HashSet, HashMap};
use std::hash::{Hash, Hasher};
use std::f64;

use super::LabeledPoint;

#[derive(Debug)]
struct MorseData {
    lifetime: f64,
    merge_parent: Option<NodeIndex>,
    ancestor: NodeIndex  // TODO: I dunno what the "proper" name for this is
}

#[derive(Debug)]
struct MorseNode {
    node: NodeIndex,
    ascending_data: Option<MorseData>,
    descending_data: Option<MorseData>
}

impl MorseNode {
    fn new(node: NodeIndex) -> MorseNode {
        MorseNode{node, ascending_data: None, descending_data: None}
    }

    fn set_data(&mut self, data: MorseData, kind: MorseKind) {
        match kind {
            MorseKind::Ascending => self.ascending_data = Some(data),
            MorseKind::Descending => self.descending_data = Some(data)
        }
    }

    fn get_data(&self, kind: MorseKind) -> &Option<MorseData> {
        match kind {
            MorseKind::Ascending => &self.ascending_data,
            MorseKind::Descending => &self.descending_data
        }
    }
}

impl Hash for MorseNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.node.hash(state);
    }
}

impl PartialEq for MorseNode {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node
    }
}

impl Eq for MorseNode {}

struct PointedUnionFind {
    unionfind: UnionFind<usize>,
    reprs: Vec<usize>
}

impl PointedUnionFind {
    // this is insanely specific and will break if you use it outside of exactly
    // how it works in the morse complex code (and maybe even if you use it
    // exactly that way!)
    // This turns UnionFind into a structure that always keeps the representative
    // for the left hand size of a union constant. But to do this O(1)
    // i can't do things like "ensure consistency" outside of the access patterns
    // i know the morse complex code will follow
    // (specifically, this data structure offers no guarantees that
    // `find(find(x)) will be reasonable)
    fn new(n: usize) -> Self {
        let unionfind = UnionFind::new(n);
        let reprs = (0..n).collect();
        PointedUnionFind{unionfind, reprs}
    }

    fn find(&self, x: usize) -> usize {
        let inner_repr = self.unionfind.find(x);
        self.reprs[inner_repr]
    }

    fn union(&mut self, x: usize, y: usize) {
        // x is privileged!
        let old_outer = self.find(x);
        self.unionfind.union(x, y);
        let new_inner = self.unionfind.find(x);
        self.reprs[new_inner] = old_outer;
    }
}

pub struct MorseComplex<'a> {
    ascending_crystals: PointedUnionFind,
    descending_crystals: PointedUnionFind,
    ordered_points: Vec<MorseNode>,
    inverse_lookup: HashMap<NodeIndex, usize>,
    graph: &'a mut Graph<LabeledPoint, f64, petgraph::Undirected>
}

#[derive(Debug, Clone, Copy)]
pub enum MorseKind {
    Ascending,
    Descending
}

impl<'a> MorseComplex<'a> {
    pub fn from_graph(graph: &'a mut Graph<LabeledPoint, f64, petgraph::Undirected>) -> MorseComplex<'a> {
        let ordered_points = MorseComplex::get_descending_nodes(graph);
        let inverse_lookup: HashMap<NodeIndex, usize> = ordered_points.iter().enumerate()
            .map(|x| (x.1.node, x.0))
            .collect();
        let num_points = ordered_points.len();
        let ascending_crystals = PointedUnionFind::new(num_points);
        let descending_crystals = PointedUnionFind::new(num_points);

        MorseComplex{ascending_crystals, descending_crystals, inverse_lookup, ordered_points, graph}
    }


    fn get_descending_nodes(graph: &Graph<LabeledPoint, f64, petgraph::Undirected>) -> Vec<MorseNode> {
        let mut nodes: Vec<NodeIndex> = graph.node_indices().collect();
        nodes.sort_by(|a, b| {
                let a_node = graph.node_weight(*a).expect("Node a wasn't in graph");
                let b_node = graph.node_weight(*b).expect("Node b wasn't in graph");
                b_node.value.partial_cmp(&a_node.value).expect("Nan in the values")
            });
        nodes.iter().enumerate().map(|(_, n)| MorseNode::new(*n)).collect()
    }

    // FIXME: once things get ironed out a bit more this should really be its own type
    pub fn get_filtration(&self, kind: MorseKind) -> Vec<(f64, NodeIndex, NodeIndex)> {
        let mut filtration = self.ordered_points.iter() 
            // FIXME: get rid of this unwrap
            .filter_map(|point| {
                let data = point.get_data(kind).as_ref().unwrap();
                if let Some(parent) = data.merge_parent {
                    Some((data.lifetime, point.node, parent))
                } else {
                    None
                }
             })
             .collect::<Vec<_>>();
        filtration.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        filtration
    }

    pub fn get_complex(&self, kind: MorseKind) -> Vec<(NodeIndex, NodeIndex)> {
        self.ordered_points.iter() 
            // FIXME: get rid of this unwrap
            .map(|point| {
                let data = point.get_data(kind).as_ref().unwrap();
                (point.node, data.ancestor)
             })
             .collect()
    }

    pub fn get_persistence(&self, kind: MorseKind) -> Option<HashMap<NodeIndex, f64>> {
        let mut result = HashMap::with_capacity(self.ordered_points.len());
        for morse_node in self.ordered_points.iter() {
            if let Some(data) = morse_node.get_data(kind) {
                result.insert(morse_node.node, data.lifetime);
            } else{
                return None;
            }
        }
        Some(result)
    }

    pub fn compute_morse_complex(&mut self, kind: MorseKind) -> &Self{
        // We iterate through the points in descending order, which means we are
        // essentially building the morse complex at the same time that we compute
        // persistence.

        // TODO: This feels like a bad way to handle the reversible order
        let order: Vec<usize> = match kind {
            MorseKind::Ascending => (0..self.ordered_points.len()).rev().collect(),
            MorseKind::Descending => (0..self.ordered_points.len()).collect(),
        };
        for i in order {
            // find all *already processed* points that we have an edge to
            let this_value = self.graph.node_weight(self.ordered_points[i].node).unwrap().value;
            let higher_indices: Vec<usize> = self.graph.neighbors(self.ordered_points[i].node)
                .filter(|n| { match kind {
                        MorseKind::Ascending => self.graph.node_weight(*n).unwrap().value <= this_value,
                        MorseKind::Descending => self.graph.node_weight(*n).unwrap().value >= this_value
                    }
                })
                .map(|n| *self.inverse_lookup.get(&n).unwrap())
                .filter(|&n_idx| { match kind {
                        MorseKind::Ascending => n_idx > i,
                        MorseKind::Descending => n_idx < i
                }})
                .collect();

            // Nothing to do if we have no neighbors, but if we do then we
            // have to merge the correspond morse crystals
            let lifetime = if higher_indices.is_empty () {
                f64::INFINITY  // FIXME: don't really like handling lifetimes this way
            } else {
                0.
            };
            let ancestor = self.merge_crystals(i, &higher_indices, kind);

            // this is not a maximum so it has no lifetime
            self.ordered_points[i].set_data(MorseData{lifetime, ancestor, merge_parent: None}, kind);
        }
        self

        // By definition, highest max has infinite persistence
        // FIXME: for now I'm just letting None indicate this. Not ideal, probably
        //self.ordered_points[0].lifetime = Some(f64::INFINITY);

        //self.ordered_points.iter()
        //    .map(|morse_node| (morse_node.node, morse_node.lifetime.expect("no lifetime?")))
        //    .collect()
    }

    fn union_crystals(&mut self, owning_index: usize, new_index: usize, kind: MorseKind) {
        match kind {
            MorseKind::Ascending => self.ascending_crystals.union(owning_index, new_index),
            MorseKind::Descending => self.descending_crystals.union(owning_index, new_index)
        }
    }

    fn find_in_crystals(&self, index: usize, kind: MorseKind) -> usize {
        match kind {
            MorseKind::Ascending => self.ascending_crystals.find(index),
            MorseKind::Descending => self.descending_crystals.find(index)
        }
    }

    // FIXME: I don't like this signature. Not at all clear what this returned nodeindex means
    fn merge_crystals(&mut self, ordered_index: usize, ascending_neighbors: &[usize], kind: MorseKind) -> NodeIndex {
        // If there are no neighbors, there's nothing to merge
        // FIXME: "ascending_neighbors" is a bad name
        if ascending_neighbors.is_empty() {
            return self.ordered_points[ordered_index].node;
        }

        // one neighbor is easy, just union this point in to that neighbor's crystal
        if ascending_neighbors.len() == 1 {
            let neighbor_index = ascending_neighbors[0];
            self.union_crystals(neighbor_index, ordered_index, kind);
            return self.ordered_points[neighbor_index].get_data(kind).as_ref().expect("Steepest neighbor had no data").ancestor;
        }

        // for multiple neighbors, first figure out if all neighbors are in the same crystal
        let connected_crystals: HashSet<_> = ascending_neighbors.iter()
            .map(|&idx| self.find_in_crystals(idx, kind))
            .collect();

        // If they are all in the same crystal, it's the same as if there was just one neighbor
        if connected_crystals.len() == 1 {
            let neighbor_index = ascending_neighbors[0];
            self.union_crystals(neighbor_index, ordered_index, kind);
            return self.ordered_points[neighbor_index].get_data(kind).as_ref().expect("Steepest neighbor had no data").ancestor;
        }

        // And if we're here then we're merging crystals
        // first figure out what the global max is
        let (_, max_crystal) = connected_crystals.iter()
            .map(|&idx| {
                let node = &self.ordered_points[idx];
                let value = self.graph.node_weight(node.node).expect("max wasn't in the graph").value;
                (value, idx)
            })
            .max_by(|a, b| a.0.partial_cmp(&b.0).expect("Nan in the values"))
            .expect("No maximum was found, somehow?");

        let joining_node = &self.ordered_points[ordered_index];

        let (_, steepest_neighbor) = ascending_neighbors.iter()
            .map(|&idx| {
                let node = &self.ordered_points[idx];
                let value = self.graph.node_weight(node.node).unwrap().value;
                let edge = self.graph.find_edge(joining_node.node, node.node).expect("A neighbor wasn't really a neighbor");
                // unwrap_or here because hte persistence calculation is still meaningful
                // even we we aren't in a metric space, so lack of grade information shouldn't
                // block the computation
                (value / *self.graph.edge_weight(edge).unwrap_or(&1.), idx)
            })
            .max_by(|a, b| match kind {
                MorseKind::Descending => a.0.partial_cmp(&b.0).expect("Nan in the values"),
                MorseKind::Ascending => b.0.partial_cmp(&a.0).expect("Nan in the values")
            })
            .expect("No steepset neighbor was found, somehow?");

        // now we need to update the lifetimes and merge the other crystals
        let joining_value = self.graph.node_weight(joining_node.node).expect("joining node wasn't in the graph").value;
        let merge_parent = self.ordered_points[max_crystal].node;
        self.union_crystals(max_crystal, ordered_index, kind);
        for crystal in connected_crystals {
            if crystal != max_crystal {
                let crystal_node = &self.ordered_points[crystal];
                let crystal_value = self.graph.node_weight(crystal_node.node).expect("crystal node wasn't in the graph").value;
                let ancestor = self.ordered_points[crystal].get_data(kind).as_ref().expect("crystal had no data").ancestor;
                let lifetime = crystal_value - joining_value;
                self.ordered_points[crystal].set_data(MorseData{ancestor, lifetime, 
                    merge_parent: Some(merge_parent)}, kind);
                self.union_crystals(max_crystal, crystal, kind);
            }
        }
        // TODO: I don't entirely understand why i need as_ref here
        self.ordered_points[steepest_neighbor].get_data(kind).as_ref().expect("steepest neighbor had no data").ancestor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;

    #[test]
    fn test_single() {
        let mut graph = Graph::new_undirected();
        let points = [
            LabeledPoint{id: 0, value: -1., point: vec![0., 0.]},
            LabeledPoint{id: 1, value: 1., point: vec![1., 0.]},
        ];
        let mut node_lookup = Vec::with_capacity(points.len());
        for point in &points {
            let node = graph.add_node(point.to_owned());
            node_lookup.push(node);
        }
        graph.add_edge(node_lookup[0], node_lookup[1], 0.);
        let mut complex = MorseComplex::from_graph(&mut graph);
        let lifetimes = complex
            .compute_morse_complex(MorseKind::Descending)
            .get_persistence(MorseKind::Descending)
            .expect("couldn't get lifetimes");
        assert_eq!(lifetimes[&node_lookup[0]], 0.);
        assert_eq!(lifetimes[&node_lookup[1]], f64::INFINITY);
    }

    #[test]
    fn test_triangle() {
        let mut graph = Graph::new_undirected();
        let points = [
            LabeledPoint{id: 0, value: -1., point: vec![0., 0.]},
            LabeledPoint{id: 1, value: 0., point: vec![1., 1.]},
            LabeledPoint{id: 2, value: 1., point: vec![1., 0.]},
        ];
        let mut node_lookup = Vec::with_capacity(points.len());
        for point in &points {
            let node = graph.add_node(point.to_owned());
            node_lookup.push(node);
        }
        graph.add_edge(node_lookup[0], node_lookup[1], 0.);
        graph.add_edge(node_lookup[0], node_lookup[2], 0.);
        graph.add_edge(node_lookup[1], node_lookup[2], 0.);
        let mut complex = MorseComplex::from_graph(&mut graph);
        let lifetimes = complex
            .compute_morse_complex(MorseKind::Descending)
            .get_persistence(MorseKind::Descending)
            .expect("couldn't get lifetimes");
        assert_eq!(lifetimes[&node_lookup[0]], 0.);
        assert_eq!(lifetimes[&node_lookup[1]], 0.);
        assert_eq!(lifetimes[&node_lookup[2]], f64::INFINITY);
    }

    #[test]
    fn test_square() {
        let mut graph = Graph::new_undirected();
        let points = [
            LabeledPoint{id: 0, value: 1., point: vec![0., 0.]},
            LabeledPoint{id: 1, value: -1., point: vec![1., 0.]},
            LabeledPoint{id: 2, value: 0., point: vec![0., 1.]},
            LabeledPoint{id: 3, value: 2., point: vec![1., 1.]},
        ];
        let mut node_lookup = Vec::with_capacity(points.len());
        for point in &points {
            let node = graph.add_node(point.to_owned());
            node_lookup.push(node);
        }
        graph.add_edge(node_lookup[0], node_lookup[1], 0.);
        graph.add_edge(node_lookup[0], node_lookup[2], 0.);
        graph.add_edge(node_lookup[1], node_lookup[3], 0.);
        graph.add_edge(node_lookup[2], node_lookup[3], 0.);
        let mut complex = MorseComplex::from_graph(&mut graph);
        let lifetimes = complex
            .compute_morse_complex(MorseKind::Descending)
            .get_persistence(MorseKind::Descending)
            .expect("couldn't get lifetimes");
        assert_eq!(lifetimes[&node_lookup[0]], 1.);
        assert_eq!(lifetimes[&node_lookup[1]], 0.);
        assert_eq!(lifetimes[&node_lookup[2]], 0.);
        assert_eq!(lifetimes[&node_lookup[3]], f64::INFINITY);
    }

    #[test]
    fn test_big_square() {
        let mut graph = Graph::new_undirected();
        let points = [
            LabeledPoint{id: 0, value: 6., point: vec![0., 0.]},
            LabeledPoint{id: 1, value: 2., point: vec![1., 0.]},
            LabeledPoint{id: 2, value: 3., point: vec![2., 0.]},
            LabeledPoint{id: 3, value: 5., point: vec![0., 1.]},
            LabeledPoint{id: 4, value: 4., point: vec![1., 1.]},
            LabeledPoint{id: 5, value: -5., point: vec![1., 2.]},
            LabeledPoint{id: 6, value: 0., point: vec![0., 2.]},
            LabeledPoint{id: 7, value: 1., point: vec![1., 2.]},
            LabeledPoint{id: 8, value: 10., point: vec![2., 2.]},
        ];
        let mut node_lookup = Vec::with_capacity(points.len());
        for point in &points {
            let node = graph.add_node(point.to_owned());
            node_lookup.push(node);
        }
        graph.add_edge(node_lookup[0], node_lookup[1], 0.);
        graph.add_edge(node_lookup[1], node_lookup[2], 0.);
        graph.add_edge(node_lookup[0], node_lookup[3], 0.);
        graph.add_edge(node_lookup[1], node_lookup[4], 0.);
        graph.add_edge(node_lookup[2], node_lookup[5], 0.);
        graph.add_edge(node_lookup[3], node_lookup[4], 0.);
        graph.add_edge(node_lookup[4], node_lookup[5], 0.);
        graph.add_edge(node_lookup[3], node_lookup[6], 0.);
        graph.add_edge(node_lookup[4], node_lookup[7], 0.);
        graph.add_edge(node_lookup[5], node_lookup[8], 0.);
        graph.add_edge(node_lookup[6], node_lookup[7], 0.);
        graph.add_edge(node_lookup[7], node_lookup[8], 0.);
        let mut complex = MorseComplex::from_graph(&mut graph);
        let lifetimes = complex
            .compute_morse_complex(MorseKind::Descending)
            .get_persistence(MorseKind::Descending)
            .expect("couldn't get lifetimes");
        assert_eq!(lifetimes[&node_lookup[0]], 5.);
        assert_eq!(lifetimes[&node_lookup[1]], 0.);
        assert_eq!(lifetimes[&node_lookup[2]], 1.);
        assert_eq!(lifetimes[&node_lookup[3]], 0.);
        assert_eq!(lifetimes[&node_lookup[4]], 0.);
        assert_eq!(lifetimes[&node_lookup[5]], 0.);
        assert_eq!(lifetimes[&node_lookup[6]], 0.);
        assert_eq!(lifetimes[&node_lookup[7]], 0.);
        assert_eq!(lifetimes[&node_lookup[8]], f64::INFINITY);
    }

    #[test]
    fn test_filtration() {
        let mut graph = Graph::new_undirected();
        let points = [
            LabeledPoint{id: 0, value: 3., point: vec![0., 0.]},
            LabeledPoint{id: 1, value: -1., point: vec![1., 0.]},
            LabeledPoint{id: 2, value: 10., point: vec![0., 1.]},
            LabeledPoint{id: 3, value: 2., point: vec![1., 1.]},
            LabeledPoint{id: 4, value: 7., point: vec![1., 1.]},
        ];
        let mut node_lookup = Vec::with_capacity(points.len());
        for point in &points {
            let node = graph.add_node(point.to_owned());
            node_lookup.push(node);
        }
        graph.add_edge(node_lookup[0], node_lookup[1], 0.);
        graph.add_edge(node_lookup[0], node_lookup[3], 0.);
        graph.add_edge(node_lookup[1], node_lookup[2], 0.);
        graph.add_edge(node_lookup[1], node_lookup[4], 0.);
        graph.add_edge(node_lookup[3], node_lookup[4], 0.);
        let mut complex = MorseComplex::from_graph(&mut graph);
        let lifetimes = complex
            .compute_morse_complex(MorseKind::Descending)
            .get_persistence(MorseKind::Descending)
            .expect("couldn't get lifetimes");
        println!("{:?}", lifetimes);
        assert_eq!(lifetimes[&node_lookup[0]], 1.);
        assert_eq!(lifetimes[&node_lookup[1]], 0.);
        assert_eq!(lifetimes[&node_lookup[2]], f64::INFINITY);
        assert_eq!(lifetimes[&node_lookup[3]], 0.);
        assert_eq!(lifetimes[&node_lookup[4]], 8.);

        let filtration = complex.get_filtration(MorseKind::Descending);
        let expected = [(1., node_lookup[0], node_lookup[4]), (8., node_lookup[4], node_lookup[2])];
        for (actual, expected) in filtration.iter().zip(expected.iter()) {
            assert_eq!(actual.0, expected.0);
            assert_eq!(actual.1, expected.1);
            assert_eq!(actual.2, expected.2);
        }
    }
}
