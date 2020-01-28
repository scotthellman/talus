use ndarray::prelude::*;
use itertools::Itertools;
use petgraph::graph::{Graph, NodeIndex};
use petgraph::unionfind::UnionFind;
use petgraph::dot::Dot;

use std::fs::File;
use std::collections::{HashSet, HashMap};
use std::f64;
use std::error::Error;
use std::io::BufReader;
use csv::StringRecord;
use std::hash::{Hash, Hasher};

#[derive(Debug)]
struct LabeledPoint {
    point: Array1<f64>,
    label: f64
}

impl LabeledPoint {
    fn from_record(record: &StringRecord) -> LabeledPoint {
        let label = record[record.len() - 1].parse::<f64>().expect("Expected a float");
        let point = record.iter()
            .take(record.len() - 1)
            .map(|v| v.parse::<f64>().expect("Expected a float"))
            .collect();
        LabeledPoint{point, label}
    }

    fn to_owned(&self) -> LabeledPoint {
        // This is basically clone? I'm just copying the name from ndarray for now
        LabeledPoint{label: self.label, point: self.point.to_owned()}
    }

    fn points_from_file(filename: &str) -> Result<Vec<LabeledPoint>, Box<dyn Error>> {
        let f = File::open(filename).expect("Unable to open file");
        let f = BufReader::new(f);
        let mut points = Vec::with_capacity(16);
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_reader(f);
        for result in rdr.records() {
            let mut record = result?;
            record.trim();
            points.push(LabeledPoint::from_record(&record));
        }
        Ok(points)
    }

    fn grade(&self, other: &LabeledPoint) -> f64{
        let diff = &self.point - &other.point;
        let distance = diff.dot(&diff).sqrt();
        let label_diff = self.label - other.label;
        label_diff / distance
    }
}

fn pairwise_distance(points: &[LabeledPoint]) -> Array2<f64> {
    let mut pairwise = Array2::zeros((points.len(), points.len()));
    for (i, row) in points.iter().enumerate() {
        for (j, other) in points[i..].iter().enumerate() {
            let j = j+i;
            let distance = if i == j {
                0.
            } else {
                let diff = &row.point - &other.point;
                diff.dot(&diff).sqrt()
            };
            pairwise[[i,j]] = distance;
            pairwise[[j,i]] = distance;
        }
    }
    pairwise
}

fn build_knn(points: &[LabeledPoint], k: usize) -> Graph<LabeledPoint, f64, petgraph::Undirected> {
    let mut neighbor_graph = Graph::new_undirected();
    let mut node_lookup = Vec::with_capacity(points.len());
    for point in points {
        let node = neighbor_graph.add_node(point.to_owned());
        node_lookup.push(node);
    }
    let pairwise = pairwise_distance(points);
    for (i, _) in points.iter().enumerate() {
        pairwise.slice(s![i, ..]).into_iter().enumerate()
            .filter(|(j, _)| i != *j)
            .sorted_by(|(_, val), (_, other)| val.partial_cmp(other).unwrap())
            .take(k)
            .for_each(|(j, val)| {
                neighbor_graph.add_edge(node_lookup[i], node_lookup[j], *val);
            });
    }
    neighbor_graph
}

fn find_steepest_neighbor(node: NodeIndex,
                          graph: &Graph<LabeledPoint, f64, petgraph::Undirected>) -> Option<NodeIndex> {
    let this_point = graph.node_weight(node).unwrap();
    let result = graph.neighbors(node)
        .map(|n_idx| (n_idx, graph.node_weight(n_idx).unwrap()))
        .filter(|(_, n)| n.label > this_point.label)
        .map(|(n_idx, n)| (n_idx, n, this_point.grade(&n)))
        .max_by(|a, b| a.2.partial_cmp(&b.2).expect("Nan in the labels"));
    match result {
        None => None,
        Some((idx, _, _)) => Some(idx)
    }
}

fn partition_graph_by_steepest_ascent(graph: &Graph<LabeledPoint, f64, petgraph::Undirected>) {
    //FIXME actually implement this
    for node in graph.node_indices() {
        let neighbor = find_steepest_neighbor(node, graph);
        println!("For {:?}, steepest neighbor was {:?}", node, neighbor);
    }
}

fn get_descending_nodes(graph: &Graph<LabeledPoint, f64, petgraph::Undirected>) -> Vec<MorseNode> {
    let mut nodes: Vec<NodeIndex> = graph.node_indices().collect();
    nodes.sort_by(|a, b| {
            let a_node = graph.node_weight(*a).expect("Node a wasn't in graph");
            let b_node = graph.node_weight(*b).expect("Node b wasn't in graph");
            b_node.label.partial_cmp(&a_node.label).expect("Nan in the labels")
        });
    nodes.iter().enumerate().map(|(i, n)| MorseNode{node: *n, lifetime:None}).collect()
}

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
    // This turns UnionFind into a structure that always keeps the repr 
    // for the left hand size of a union constant. But to do this efficiently
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

struct MorseComplex<'a> {
    crystals: PointedUnionFind,
    ordered_points: Vec<MorseNode>,
    graph: &'a mut Graph<LabeledPoint, f64, petgraph::Undirected>
}

impl<'a> MorseComplex<'a> {
    fn from_graph(graph: &'a mut Graph<LabeledPoint, f64, petgraph::Undirected>) -> MorseComplex<'a> {
        let ordered_points = get_descending_nodes(graph);
        let num_points = ordered_points.len();
        let crystals = PointedUnionFind::new(num_points);

        MorseComplex{crystals, ordered_points, graph}
    }

    fn compute_persistence(&mut self) -> HashMap<NodeIndex, f64> {
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

fn main() {
    let points = match LabeledPoint::points_from_file("points.txt") {
        Ok(points) => points,
        Err(e) => {
            println!("Failed to parse points: {}", e);
            panic!();
        }
    };
    let mut graph = build_knn(&points, 2);
    println!("Graph is {:?}", graph);
    println!("{:?}", Dot::with_config(&graph, &[]));
    let mut complex = MorseComplex::from_graph(&mut graph);
    let lifetimes = complex.compute_persistence();
    println!("Lifetimes were {:?}", lifetimes);
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single() {
        let mut graph = Graph::new_undirected();
        let points = [
            LabeledPoint{label: -1., point: arr1(&[0., 0.])},
            LabeledPoint{label: 1., point: arr1(&[1., 0.])},
        ];
        let mut node_lookup = Vec::with_capacity(points.len());
        for point in &points {
            let node = graph.add_node(point.to_owned());
            node_lookup.push(node);
        }
        graph.add_edge(node_lookup[0], node_lookup[1], 0.);
        let mut complex = MorseComplex::from_graph(&mut graph);
        let lifetimes = complex.compute_persistence();
        assert_eq!(lifetimes[&node_lookup[0]], 0.);
        assert_eq!(lifetimes[&node_lookup[1]], f64::INFINITY);
    }

    #[test]
    fn test_triangle() {
        let mut graph = Graph::new_undirected();
        let points = [
            LabeledPoint{label: -1., point: arr1(&[0., 0.])},
            LabeledPoint{label: 0., point: arr1(&[1., 1.])},
            LabeledPoint{label: 1., point: arr1(&[1., 0.])},
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
        let lifetimes = complex.compute_persistence();
        assert_eq!(lifetimes[&node_lookup[0]], 0.);
        assert_eq!(lifetimes[&node_lookup[1]], 0.);
        assert_eq!(lifetimes[&node_lookup[2]], f64::INFINITY);
    }

    #[test]
    fn test_square() {
        let mut graph = Graph::new_undirected();
        let points = [
            LabeledPoint{label: 1., point: arr1(&[0., 0.])},
            LabeledPoint{label: -1., point: arr1(&[1., 0.])},
            LabeledPoint{label: 0., point: arr1(&[0., 1.])},
            LabeledPoint{label: 2., point: arr1(&[1., 1.])},
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
        let lifetimes = complex.compute_persistence();
        assert_eq!(lifetimes[&node_lookup[0]], 1.);
        assert_eq!(lifetimes[&node_lookup[1]], 0.);
        assert_eq!(lifetimes[&node_lookup[2]], 0.);
        assert_eq!(lifetimes[&node_lookup[3]], f64::INFINITY);
    }

    #[test]
    fn test_big_square() {
        let mut graph = Graph::new_undirected();
        let points = [
            LabeledPoint{label: 6., point: arr1(&[0., 0.])},
            LabeledPoint{label: 2., point: arr1(&[1., 0.])},
            LabeledPoint{label: 3., point: arr1(&[2., 0.])},
            LabeledPoint{label: 5., point: arr1(&[0., 1.])},
            LabeledPoint{label: 4., point: arr1(&[1., 1.])},
            LabeledPoint{label: -5., point: arr1(&[1., 2.])},
            LabeledPoint{label: 0., point: arr1(&[0., 2.])},
            LabeledPoint{label: 1., point: arr1(&[1., 2.])},
            LabeledPoint{label: 10., point: arr1(&[2., 2.])},
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
        let lifetimes = complex.compute_persistence();
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
}
