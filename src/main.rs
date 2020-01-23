use ndarray::prelude::*;
use itertools::Itertools;
use petgraph::graph::{Graph, NodeIndex};
use petgraph::dot::Dot;

use std::fs::File;
use std::io;
use std::process;
use std::error::Error;
use std::io::{BufRead, BufReader};
use csv::StringRecord;
use partitions::partition_vec::PartitionVec;

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

fn get_descending_nodes(graph: &Graph<LabeledPoint, f64, petgraph::Undirected>) -> PartitionVec<NodeIndex> {
    let mut nodes: PartitionVec<NodeIndex> = graph.node_indices().collect();
    nodes.sort_by(|a, b| {
            let a_node = graph.node_weight(*a).expect("Node a wasn't in graph");
            let b_node = graph.node_weight(*b).expect("Node b wasn't in graph");
            b_node.label.partial_cmp(&a_node.label).expect("Nan in the labels")
        });
    nodes
}

// TODO: I'm not convinced PartitionVec is helping me. I need a pointed set and
// that isn't what it gives me. I need a new struct, something like

struct MorseNode {
    node: NodeIndex,
    maxima: NodeIndex
}

// This signature sucks so much
fn merge_crystals(node: &NodeIndex, list_index: usize, higher_neighbors: &Vec<(usize, &NodeIndex)>,
                  ordered_nodes: &mut PartitionVec<NodeIndex>,
                  lifetimes: &mut Vec<f64>, graph: &Graph<LabeledPoint, f64, petgraph::Undirected>) {
    // guard against the no-neighbor case
    if higher_neighbors.is_empty() {
        return;
    }
    // simplest case: one neighbor
    if higher_neighbors.len() == 1 {
        let neighbor_index = higher_neighbors[0].0;
        ordered_nodes.union(list_index, neighbor_index);
        return;
    }

    // slightly worse: multiple neighbors, but all same set
    // perhaps what should be done here is: find the maximum for each
    let neighbors_differ = higher_neighbors.iter().fold( (true, None), |acc, x| {

    });
}

fn compute_persistence(graph: &Graph<LabeledPoint, f64, petgraph::Undirected>) {
    let mut ordered_nodes = get_descending_nodes(graph);
    let mut lifetimes = Vec::with_capacity(ordered_nodes.len());
    // for each node, see if it is connected to any other nodes already revealed
    // if not, then it is a maximum, i'll need to log that somehow
    // if so, then it is not a maximum. union with its neighbor
    // if there's more than one edge, and they belong to the same maximum, keep going
    // if they unite different maxima, then union the lower one(s) and log the lifetimes
    for (i, &node) in ordered_nodes.iter().enumerate() {
        let labeled_point = graph.node_weight(node).expect("Node wasn't in graph");
        let higher_neighbors: Vec<_> = ordered_nodes.iter().enumerate()
            .take(i)
            .filter(|(j, &n_idx)| graph.find_edge(node, n_idx).is_some())
            .collect();

        if higher_neighbors.is_empty() {
            // this is a maximum
            lifetimes.push(labeled_point.label);
        } else {
            // this is not a maximum so:
            // it has no lifetime
            lifetimes.push(0.);

            // now handle whatever merging we need
            merge_crystals(&node, i, &higher_neighbors, &mut ordered_nodes, &mut lifetimes, graph);
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
    let graph = build_knn(&points, 2);
    println!("Graph is {:?}", graph);
    println!("{:?}", Dot::with_config(&graph, &[]));
    partition_graph_by_steepest_ascent(&graph);
}
