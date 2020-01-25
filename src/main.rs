use ndarray::prelude::*;
use itertools::Itertools;
use petgraph::graph::{Graph, NodeIndex};
use petgraph::dot::Dot;

use std::fs::File;
use std::collections::{HashSet, HashMap};
use std::io;
use std::f64;
use std::process;
use std::error::Error;
use std::io::{BufRead, BufReader};
use csv::StringRecord;

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
    nodes.iter().enumerate().map(|(i, n)| MorseNode{node: *n, maximum:None, maximum_idx: None, node_idx: i}).collect()
}

#[derive(PartialEq, Eq, Hash, Debug)]
struct MorseNode {
    node: NodeIndex,
    maximum: Option<NodeIndex>,
    maximum_idx: Option<usize>, //FIXME: i hate this
    node_idx: usize //FIXME this too

}

// This signature sucks so much
fn merge_crystals(list_index: usize, higher_indices: &Vec<usize>, ordered_nodes: &mut Vec<MorseNode>,
                  lifetimes: &mut Vec<f64>, graph: &Graph<LabeledPoint, f64, petgraph::Undirected>) {
    // guard against the no-neighbor case
    if higher_indices.is_empty() {
        return;
    }
    // simplest case: one neighbor
    if higher_indices.len() == 1 {
        let neighbor_index = higher_indices[0];
        ordered_nodes[list_index].maximum = ordered_nodes[neighbor_index].maximum;
        ordered_nodes[list_index].maximum_idx = ordered_nodes[neighbor_index].maximum_idx;
        return;
    }

    // two cases for multiple neighbors. all the same (no merge), or some different (merge)
    println!("{:?}", higher_indices);
    let all_maxima_indices: HashSet<usize> = higher_indices.iter().map(|i| {
            ordered_nodes[*i].maximum_idx.expect("somehow a node in higher_indices had no maximum")
        }).collect();
    if all_maxima_indices.len() == 1 {
        let maximum_idx = all_maxima_indices.iter().next().unwrap();
        ordered_nodes[list_index].maximum = Some(ordered_nodes[*maximum_idx].node);
        ordered_nodes[list_index].maximum_idx = Some(*maximum_idx);
    } else {
        // and this is the tough one. We need to identify the highest maxima,
        // change everything to the maxima, and _propagate that change_
        let (_, max_idx) = all_maxima_indices.iter().fold((f64::NEG_INFINITY, None), |acc, max_idx| {
            let node = ordered_nodes[*max_idx].node;
            let value = graph.node_weight(node).expect("max wasn't in the graph").label;
            if value > acc.0{
                (value, Some(max_idx))
            } else {
                acc
            }
        });
        let max_idx = *max_idx.expect("Somehow we have no maximum even though there were neighbors");
        for &local_idx in &all_maxima_indices {
            if local_idx != max_idx {
                // This is the critical bit. We subtract the max's value from the 
                // value of the node that connected the two crystals
                // FIXME: i really should not be overloading lifetimes like this
                lifetimes[local_idx] = lifetimes[max_idx] - lifetimes[list_index];
            }
        }
        // FIXME: This all sucks
        let max_node = ordered_nodes[max_idx].maximum;
        for mut node in ordered_nodes {
            if let Some(idx) = node.maximum_idx {
                if all_maxima_indices.contains(&idx) {
                    node.maximum = max_node;
                    node.maximum_idx = Some(max_idx);
                }
            }
        }

    }
}
// wow that's some bad code

fn compute_persistence(graph: &Graph<LabeledPoint, f64, petgraph::Undirected>) -> HashMap<NodeIndex, f64> {
    let mut ordered_nodes = get_descending_nodes(graph);
    let mut lifetimes = Vec::with_capacity(ordered_nodes.len());
    // for each node, see if it is connected to any other nodes already revealed
    // if not, then it is a maximum, i'll need to log that somehow
    // if so, then it is not a maximum. union with its neighbor
    // if there's more than one edge, and they belong to the same maximum, keep going
    // if they unite different maxima, then union the lower one(s) and log the lifetimes
    for i in 0..ordered_nodes.len() {
        println!("{:?}", ordered_nodes);
        let labeled_point = graph.node_weight(ordered_nodes[i].node).expect("Node wasn't in graph");
        let higher_indices: Vec<_> = ordered_nodes.iter().enumerate()
            .take(i)
            .filter(|(_, neighbor)| graph.find_edge(ordered_nodes[i].node, neighbor.node).is_some())
            .map(|(j, _)| j)
            .collect();

        if higher_indices.is_empty() {
            // this is a maximum
            lifetimes.push(labeled_point.label);
            ordered_nodes[i].maximum = Some(ordered_nodes[i].node);
            ordered_nodes[i].maximum_idx = Some(i);
        } else {
            // this is not a maximum so:
            // it has no lifetime
            lifetimes.push(0.);

            // now handle whatever merging we need
            merge_crystals(i, &higher_indices, &mut ordered_nodes, &mut lifetimes, graph);
        }
        println!("{:?}", lifetimes);
    }

    // By definition, highest max has infinite persistence
    lifetimes[0] = f64::INFINITY;

    // not really sure what i want the return type to be
    ordered_nodes.iter()
        .map(|morse_node| morse_node.node)
        .zip(lifetimes.into_iter())
        .collect()
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
    let lifetimes = compute_persistence(&graph);
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
        let lifetimes = compute_persistence(&graph);
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
        let lifetimes = compute_persistence(&graph);
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
        let lifetimes = compute_persistence(&graph);
        assert_eq!(lifetimes[&node_lookup[0]], 2.);
        assert_eq!(lifetimes[&node_lookup[1]], 0.);
        assert_eq!(lifetimes[&node_lookup[2]], 0.);
        assert_eq!(lifetimes[&node_lookup[3]], f64::INFINITY);
    }
}
