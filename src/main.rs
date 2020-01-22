use ndarray::prelude::*;
use itertools::Itertools;
use petgraph::graph::Graph;
use petgraph::dot::Dot;

use std::fs::File;
use std::io;
use std::process;
use std::error::Error;
use std::io::{BufRead, BufReader};
use csv::StringRecord;

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
}

fn pairwise_distance(points: ArrayView2<f64>) -> Array2<f64> {
    let mut pairwise = Array2::zeros((points.shape()[0], points.shape()[0]));
    for (i, row) in points.outer_iter().enumerate() {
        for (j, other) in points.slice(s![i.., ..]).outer_iter().enumerate() {
            let j = j+i;
            let distance = if i == j {
                0.
            } else {
                let diff = &row - &other;
                diff.dot(&diff).sqrt()
            };
            pairwise[[i,j]] = distance;
            pairwise[[j,i]] = distance;
        }
    }
    pairwise
}

fn build_knn(points: ArrayView2<f64>, k: usize) -> Graph<Array1<f64>, f64, petgraph::Undirected> {
    let mut neighbor_graph = Graph::new_undirected();
    let mut node_lookup = Vec::with_capacity(points.shape()[0]);
    for point in points.outer_iter() {
        let node = neighbor_graph.add_node(point.to_owned());
        node_lookup.push(node);
    }
    let pairwise = pairwise_distance(points);
    for (i, _) in points.outer_iter().enumerate() {
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

fn build_matrix_from_points(points: &[LabeledPoint]) -> Array2<f64> {
    // TODO: Error handling
    let shape = (points.len(), points[0].point.len());
    let mut arr = Array2::zeros(shape);
    for (i, mut row) in arr.axis_iter_mut(Axis(0)).enumerate() {
        row.assign(&points[i].point);
    }
    arr
}

fn main() {
    let points = match LabeledPoint::points_from_file("points.txt") {
        Ok(points) => points,
        Err(e) => {
            println!("Failed to parse points: {}", e);
            panic!();
        }
    };
    let points = build_matrix_from_points(&points);
    let pairwise = pairwise_distance(points.view());
    println!("Pairwise is {:?}", pairwise);
    let graph = build_knn(points.view(), 2);
    println!("Graph is {:?}", graph);
    println!("{:?}", Dot::with_config(&graph, &[]));
}
