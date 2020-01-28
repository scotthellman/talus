use petgraph::graph::{Graph, NodeIndex};
use petgraph::unionfind::UnionFind;

use std::collections::{HashSet, HashMap};
use std::hash::{Hash, Hasher};
use std::f64;

use super::LabeledPoint;

#[derive(Debug)]
struct MorseNode {
    node: NodeIndex,
    lifetime: Option<f64>,
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
    crystals: PointedUnionFind,
    ordered_points: Vec<MorseNode>,
    graph: &'a mut Graph<LabeledPoint, f64, petgraph::Undirected>
}

impl<'a> MorseComplex<'a> {
    pub fn from_graph(graph: &'a mut Graph<LabeledPoint, f64, petgraph::Undirected>) -> MorseComplex<'a> {
        let ordered_points = MorseComplex::get_descending_nodes(graph);
        let num_points = ordered_points.len();
        let crystals = PointedUnionFind::new(num_points);

        MorseComplex{crystals, ordered_points, graph}
    }

    fn get_descending_nodes(graph: &Graph<LabeledPoint, f64, petgraph::Undirected>) -> Vec<MorseNode> {
        let mut nodes: Vec<NodeIndex> = graph.node_indices().collect();
        nodes.sort_by(|a, b| {
                let a_node = graph.node_weight(*a).expect("Node a wasn't in graph");
                let b_node = graph.node_weight(*b).expect("Node b wasn't in graph");
                b_node.label.partial_cmp(&a_node.label).expect("Nan in the labels")
            });
        nodes.iter().enumerate().map(|(_, n)| MorseNode{node: *n, lifetime:None}).collect()
    }

    pub fn compute_persistence(&mut self) -> HashMap<NodeIndex, f64> {
        for i in 0..self.ordered_points.len() {
            let higher_indices: Vec<_> = self.ordered_points.iter().enumerate()
                .take(i)
                .filter(|(_, neighbor)| self.graph.find_edge(self.ordered_points[i].node, neighbor.node).is_some())
                .map(|(j, _)| j)
                .collect();

            if !higher_indices.is_empty() {
                // this is not a maximum so:
                // it has no lifetime
                self.ordered_points[i].lifetime = Some(0.);

                // now handle whatever merging we need
                self.merge_crystals(i, &higher_indices);
            }
        }

        // By definition, highest max has infinite persistence
        self.ordered_points[0].lifetime = Some(f64::INFINITY);

        self.ordered_points.iter()
            .map(|morse_node| (morse_node.node, morse_node.lifetime.expect("no lifetime?")))
            .collect()
    }

    fn merge_crystals(&mut self, ordered_index: usize, ascending_neighbors: &[usize]) {

        // guard against no neighbors
        if ascending_neighbors.is_empty() {
            return;
        }

        // one neighbor is easy
        if ascending_neighbors.len() == 1 {
            let neighbor_index = ascending_neighbors[0];
            self.crystals.union(neighbor_index, ordered_index);
            return;
        }

        // figure out if all neighbors are in the same crystal
        let connected_crystals: HashSet<_> = ascending_neighbors.iter()
            .map(|&idx| self.crystals.find(idx))
            .collect();

        if connected_crystals.len() == 1 {
            let neighbor_index = ascending_neighbors[0];
            self.crystals.union(neighbor_index, ordered_index);
            return;
        }

        // ok, if we're here then we're merging crystals
        // first figure out what the global max is
        let (_, max_crystal) = connected_crystals.iter()
            .map(|&idx| {
                let node = &self.ordered_points[idx];
                let value = self.graph.node_weight(node.node).expect("max wasn't in the graph").label;
                (value, idx)
            })
            .max_by(|a, b| a.0.partial_cmp(&b.0).expect("Nan in the labels"))
            .expect("No maximum was found, somehow?");

        // now we need to update the lifetimes and merge the other crystals
        let joining_node = &self.ordered_points[ordered_index];
        let joining_label = self.graph.node_weight(joining_node.node).expect("joining node wasn't in the graph").label;
        self.crystals.union(max_crystal, ordered_index);
        for crystal in connected_crystals {
            if crystal != max_crystal {
                //update lifetime
                let crystal_node = &self.ordered_points[crystal];
                let crystal_label = self.graph.node_weight(crystal_node.node).expect("crystal node wasn't in the graph").label;
                self.ordered_points[crystal].lifetime = Some(crystal_label - joining_label);

                //union
                self.crystals.union(max_crystal, crystal);
            }
        }
    }
}
