use talus;
use std::f64;
use std::path::PathBuf;
use std::collections::HashMap;

#[test]
fn continental_divide_test() {
    let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("tests");
    d.push("resources");
    d.push("grays.txt");
    let points = talus::LabeledPoint::points_from_file(d).ok().unwrap();
    // I know that Grays, Torreys, and Grizzly are the first 3 points
    let mut expected = HashMap::with_capacity(3);
    expected.insert(0, f64::INFINITY);
    expected.insert(1, 560.);
    expected.insert(2, 827.);
    let graph = talus::graph::build_knn(&points, 5);
    let complex = talus::morse::MorseSmaleComplex::from_graph(&graph);
    let lifetimes = complex.descending_complex.get_persistence();
    lifetimes.iter()
        .map(|(node, lifetime)| (graph.node_weight(*node).unwrap().id, lifetime))
        .filter(|(id, _)| expected.contains_key(id))
        .for_each(|(id, lifetime)| {
            let expected_lifetime = expected.get(&id).unwrap();
            println!("{}, {}, {}", id, lifetime, expected_lifetime);
            if lifetime.is_infinite() {
                assert!(expected_lifetime.is_infinite());
            } else {
                // big error bars on this, due to the manual sampling of the points in grays.txt
                assert!((lifetime - expected_lifetime).abs() < 150.);
            }
        });
}
