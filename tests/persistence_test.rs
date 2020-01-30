use talus;

#[test]
fn continental_divide_test() {
    let points = talus::LabeledPoint::points_from_file("grays.txt").ok().unwrap();
    let mut graph = talus::graph::build_knn(&points, 5);
    let mut complex = talus::morse::MorseComplex::from_graph(&mut graph);
    let lifetimes = complex.compute_persistence();
    println!("{:?}", lifetimes);
    panic!();
}
