use itertools::Itertools;
use itertools::FoldWhile::{Continue, Done};
use super::simplex::{Simplex, CNS, SimplexConverter};
use std::collections::{HashSet, HashMap};
use super::rips::RichSimplex;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct PersistencePair {
    face_index: usize,
    coface_index: usize
}

struct HomologyData {
    simplex: Simplex,
    marked: bool,
    value: Option<usize>,
    assignment: Simplex
}


#[derive(Debug)]
struct TransformationMatrix {
    columns: HashMap<usize, HashSet<usize>>
}

impl TransformationMatrix {
    fn new(num_columns: usize) -> Self {
        let columns = HashMap::with_capacity(num_columns);
        TransformationMatrix{columns}
    }

    fn get_column(&mut self, col: usize) -> &HashSet<usize> {
        self.ensure_column_exists(col);
        self.columns.get(&col).unwrap()
    }

    fn ensure_column_exists(&mut self, col: usize) {
        if !self.columns.contains_key(&col){
            let mut new_col = HashSet::with_capacity(5); // FIXME better capacity
            new_col.insert(col);
            self.columns.insert(col, new_col);
        }
    }

    fn add_first_to_second(&mut self, col: usize, other: usize) {
        // TODO: check on how fast this is vs mutating
        self.ensure_column_exists(col);
        self.ensure_column_exists(other);

        let this_col = self.columns.get(&col).unwrap();
        let other_col = self.columns.get(&other).unwrap();


        let new_col: HashSet<usize> = this_col.symmetric_difference(other_col).copied().collect();
        self.columns.insert(other, new_col);
    }

    /*
    fn remap_rows(&mut self, index_list: &[usize]) {
        let new_columns: HashMap<usize, HashSet<usize>> = self.columns.iter()
            .map(|(&i, col)| {
                let new_col: HashSet<usize> = col.iter()
                    .map(|&row| index_list[row])
                    .collect();
                (i, new_col)
            })
            .collect();
        self.columns = new_columns;
    }
    */
    fn remap_rows(mut self, index_list: &[usize]) -> Self {
        self.columns = self.columns.into_iter()
            .map(|(k, v)| (index_list[k], v))
            .collect();
        self
    }
}

#[derive(Debug)]
struct InterimPersistenceResult {
    transformation: TransformationMatrix,
    pairs: Vec<PersistencePair>,
    essentials: Vec<usize>
}


fn update_transformation(cofaces: &mut HashSet<usize>, face_simplices: &[RichSimplex],
                        coface_indices: &HashMap<CNS, usize>, converter: &SimplexConverter,
                        transformation: &mut TransformationMatrix, target_column: usize, pivot_column: usize) {
    /* Adds pivot_column to target_column, mod 2, and updates the associated set of cofaces
    */
    println!("In apply with {:?}, {:?}, {:?}, {:?}, {:?}", cofaces, face_simplices, transformation,
            target_column, pivot_column);
    let current_transformation = transformation.get_column(pivot_column);
    println!("current transformation {:?}", current_transformation);

    for i in 0..pivot_column+1 { // Could instead check all values in transformation, esp if they were ordered somehow
        println!("face: {:}, in current: {:}", i, current_transformation.contains(&i));
        println!("cofaces {:?}", cofaces);
        if current_transformation.contains(&i) {
            for coface in face_simplices[i].cofacets(converter) {
                if let Some(coface_index) = coface_indices.get(&coface.simplex){
                        if !cofaces.remove(coface_index) {
                            cofaces.insert(*coface_index);
                    }
                }
            }
        }
    }
    println!("cofaces {:?}", cofaces);
    // face-th column of transformations needs to have the current transformation added to it
    transformation.add_first_to_second(pivot_column, target_column);
}

#[derive(Debug)]
struct CofaceInfo {
    pivot_index: Option<usize>,
    maximal: bool,
    coface_indices: HashSet<usize>
}


fn get_coface_information(face: &RichSimplex, cofaces: &[RichSimplex], coface_indices: &HashMap<CNS, usize>,
              pivots: &HashMap<usize, usize>, converter: &SimplexConverter) -> CofaceInfo {
    face.cofacets(converter)
        .filter(|c| coface_indices.contains_key(&c.simplex))
        .fold_while(CofaceInfo{pivot_index: None, maximal: false, coface_indices: HashSet::new()}, |mut acc, c| {
            let coface_index = *coface_indices.get(&c.simplex).unwrap();
            acc.coface_indices.insert(coface_index);
            if acc.maximal {
                return Continue(acc)
            } else {
                if cofaces[coface_index].lifetime == face.lifetime{
                    acc.maximal = true;
                    if !pivots.contains_key(&coface_index) {
                        println!("early abandon");
                        acc.pivot_index = Some(coface_index);
                        return Done(acc)
                    }
                }
            }
            Continue(acc)
        }).into_inner()
}

fn find_persistent_pairs(faces: &[RichSimplex], cofaces: &[RichSimplex],
                         converter: &SimplexConverter) -> InterimPersistenceResult {
    // TODO: better capacities
    let coface_indices: HashMap<CNS, usize> = cofaces.iter().enumerate().map(|(i, c)| (*&c.simplex, i)).collect();
    // Maps from coface to face indices for pivots
    let mut pivots: HashMap<usize, usize> = HashMap::with_capacity(cofaces.len());

    let mut essentials: Vec<usize> = Vec::with_capacity(cofaces.len());
    let mut pairs: Vec<PersistencePair> = Vec::with_capacity(cofaces.len());
    let mut transformation: TransformationMatrix = TransformationMatrix::new(faces.len());
    for (i, face) in faces.iter().enumerate() {
        println!("Looking at {:?}", face);
        println!("Pairs are {:?}", pairs);
        let mut pivot_result = get_coface_information(face, cofaces, &coface_indices, &pivots, converter);
        if let Some(pivot) = pivot_result.pivot_index {
            pairs.push(PersistencePair{face_index:i, coface_index: pivot});
            pivots.insert(pivot, i);
            continue;
        }
        let mut failsafe = 0;
        let mut pivot = None;
        while let Some(pivot_val) = pivot_result.coface_indices.iter().max().copied() {
            pivot = Some(pivot_val);
            println!("{:?} <- {:?}", pivot, pivot_result.coface_indices);
            failsafe += 1;
            if failsafe > 10 {
                break;
            }
            match pivots.get(&pivot_val) {
                None => {break},
                Some(pivot_face) => {
                    update_transformation(&mut pivot_result.coface_indices, faces, &coface_indices,
                                         converter, &mut transformation, i, *pivot_face);
                }
            }
        }
        if pivot_result.coface_indices.len() == 0 {
            essentials.push(i);
        } else {
            let pivot = pivot.unwrap();
            pairs.push(PersistencePair{face_index: i, coface_index: pivot});
            pivots.insert(pivot, i);
        }
    }
    InterimPersistenceResult{transformation, pairs, essentials}
}

fn find_pairs_with_clearing(faces: &[RichSimplex], cofaces: &[RichSimplex], face_pairs: &[PersistencePair],
                            reduced_faces: &HashMap<usize, Vec<usize>>, converter: &SimplexConverter) -> InterimPersistenceResult {
    let pivots: HashSet<usize> = face_pairs.iter().map(|PersistencePair{face_index: s, coface_index: f}| *f)
        .collect();
    let mut index_lookup: Vec<usize> = Vec::with_capacity(faces.len() - pivots.len());
    // TODO: Maybe don't need to copy here?
    let mut filtered_faces: Vec<RichSimplex> = Vec::with_capacity(faces.len() - pivots.len());
    for (i, f) in faces.iter().enumerate() {
        if pivots.contains(&i) {
            continue;
        }
        index_lookup.push(i);
        filtered_faces.push(f.clone());
    }

    let mut result = find_persistent_pairs(&filtered_faces, cofaces, converter);

    // Now we need to fix our indexing to correspond to the full faces
    result.transformation = result.transformation.remap_rows(&index_lookup);
    result.pairs.iter_mut().for_each(|pair| pair.face_index = index_lookup[pair.face_index]);
    for i in 0..result.essentials.len() {
        result.essentials[i] = index_lookup[result.essentials[i]];
    }

    // This is lines 7-8 from alg 2
    // FIXME: I am certain it's implemented incorrectly
    // nb translating from paper, j=s, i=f
    // so specifically:
    // for (subface, face) from previous dimension's pairs
    //    transformation[face] = R[subface]
    // what is R, you might ask? the reduced filtration. which we have to recompute.
    // we recompute it by...
    //   original boundary matrix * the transformation
    // presumably one can use update_transformation for this?

    for PersistencePair{face_index: s, coface_index: f} in face_pairs {
        // ok, reading between the lines:
        // f was some linear combination of other fs, so we have to account for that
        // and include all of them in this transformation
        // TODO: presumably, f could not already exist in transformation?
        result.transformation.columns.insert(*f, reduced_faces[s].iter().copied().collect());
    }

    result
}

fn compute_face_reductions(subfaces: &[RichSimplex], faces: &[RichSimplex],
                pairs: &[PersistencePair],
                transformation: &mut TransformationMatrix,
                converter: &SimplexConverter) -> HashMap<usize, Vec<usize>> {
    let mut face_columns: HashMap<usize, Vec<usize>> = HashMap::with_capacity(10); //FIXME again, more capacity things
    for PersistencePair{face_index: s, coface_index: f} in pairs.iter() {
        let selected = transformation.get_column(*s);
        let mut reduced: HashSet<CNS> = HashSet::with_capacity(10); // FIXME: I can definitely know this better a priori

        for face_idx in selected.iter().filter(|&&i| i < *s) {
            let face_simplex = &faces[*face_idx];
            if !reduced.remove(&face_simplex.simplex) {
                reduced.insert(face_simplex.simplex);
            }
        }
        let filtered_column: Vec<usize> = faces.iter().enumerate()
            .filter_map(|(i, f)| {
                if reduced.contains(&f.simplex) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        face_columns.insert(*f, filtered_column);
    }
    face_columns
}

fn compute_barcodes(complex: &[RichSimplex], converter: &SimplexConverter) -> Vec<Vec<(f64, f64)>> {
    let max_dimension = complex.iter().map(|s| usize::from(s.dimension)).max().unwrap();
    let mut lifetimes: Vec<Vec<(f64, f64)>> = Vec::with_capacity(max_dimension+1);
    let mut dimension_pairs: Vec<Vec<PersistencePair>> = Vec::with_capacity(max_dimension+1);
    let mut dimension_essentials: Vec<Vec<usize>> = Vec::with_capacity(max_dimension+1);
    let mut dimension_partitions: Vec<Vec<RichSimplex>> = (0..max_dimension+1).map(|_| {
            //FIXME: this capacity is pretty off
            Vec::with_capacity(complex.len() / max_dimension)
        }).collect();

    // iterate in reverse because ripser uses cohomology, so we need to go backwards
    // (but all the other machinery is built for homology, so to go backwards we need to flip
    // everything around)
    for s in complex.iter().rev() {
        dimension_partitions[usize::from(s.dimension)].push(s.clone());
    }


    // H0 has to be calculated normally 
    // TODO: it can be done faster than the general homology algorithm

    let InterimPersistenceResult {transformation, pairs, essentials} = find_persistent_pairs(&dimension_partitions[0], &dimension_partitions[1], converter);
    lifetimes.push(pairs.iter().map(|pair| (dimension_partitions[0][pair.face_index].lifetime, dimension_partitions[1][pair.coface_index].lifetime)).collect());
    dimension_pairs.push(pairs);
    dimension_essentials.push(essentials);
    let mut current_transformation = transformation;

    for d in 2..max_dimension+1 {
        let subfaces = &dimension_partitions[d-2];
        let faces = &dimension_partitions[d-1];
        let cofaces = &dimension_partitions[d];
        let pairs = &dimension_pairs[d-2];
        let reduced_faces = compute_face_reductions(&subfaces, &faces, &pairs, &mut current_transformation, &converter);
        println!("iterating up with {:?} {:?} {:?}", subfaces, faces, cofaces);

        let InterimPersistenceResult {transformation, pairs, essentials} = find_pairs_with_clearing(&faces, &cofaces, &pairs, &reduced_faces, converter);
        current_transformation = transformation;
        lifetimes.push(pairs.iter().map(|pair| (faces[pair.face_index].lifetime, cofaces[pair.coface_index].lifetime)).collect());
        dimension_pairs.push(pairs);
        dimension_essentials.push(essentials);
    }

    // TODO: how does the julia code do the barcodes?
    lifetimes
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::rips;
    use proptest::prelude::*;

    #[test]
    fn test_transformation_matrix() {
        let mut t = TransformationMatrix::new(3);

        // verify we start at I
        for i in 0..3 {
            let expected: HashSet<usize> = [i].iter().copied().collect();
            let actual = t.get_column(i);
            assert!(expected.is_subset(actual));
            assert!(expected.is_superset(actual));
        }

        // Add 2 to 1
        t.add_first_to_second(1,2);
        for i in 0..3 {
            let mut expected: HashSet<usize> = [i].iter().copied().collect();
            if i == 2 {
                expected.insert(1);
            }
            let actual = t.get_column(i);
            assert!(expected.is_subset(actual));
            assert!(expected.is_superset(actual));
        }

        // We're working mod 2 so add 2 to 1 again to get I
        t.add_first_to_second(1,2);
        for i in 0..3 {
            let expected: HashSet<usize> = [i].iter().copied().collect();
            let actual = t.get_column(i);
            assert!(expected.is_subset(actual));
            assert!(expected.is_superset(actual));
        }
    }

    #[test]
    fn test_get_coface_information() {
        let converter = SimplexConverter::construct_for_vertex_count_and_dim(3, 3);
        let cofaces = [
            RichSimplex::from_vertices(&[0, 2], 2., &converter),
            RichSimplex::from_vertices(&[1, 2], 1.5, &converter),
            RichSimplex::from_vertices(&[0, 1], 0., &converter),
        ];
        let coface_indices:HashMap<CNS, usize> = cofaces.iter().enumerate()
            .map(|(i,s)| (s.simplex, i))
            .collect();
        let pivots: HashMap<usize, usize> = HashMap::new();

        let face = RichSimplex::from_vertices(&[0], 0., &converter);
        let result = get_coface_information(&face, &cofaces, &coface_indices, &pivots, &converter);
        assert_eq!(result.pivot_index, Some(2));
        assert!(result.maximal);
        assert_eq!(result.coface_indices, [0, 2].iter().copied().collect());

        let face = RichSimplex::from_vertices(&[2], 0., &converter);
        let result = get_coface_information(&face, &cofaces, &coface_indices, &pivots, &converter);
        assert_eq!(result.pivot_index, None);
        assert!(!result.maximal);
        assert_eq!(result.coface_indices, [0, 1].iter().copied().collect());
    }

    fn get_faces_and_cofaces() -> (Vec<RichSimplex>, Vec<RichSimplex>, SimplexConverter) {
        let converter = SimplexConverter::construct_for_vertex_count_and_dim(3, 3);
        let faces = vec![
            RichSimplex::from_vertices(&[0], 0., &converter),
            RichSimplex::from_vertices(&[1], 0., &converter),
            RichSimplex::from_vertices(&[2], 0., &converter),
        ];
        let cofaces = vec![
            RichSimplex::from_vertices(&[0, 2], 2., &converter),
            RichSimplex::from_vertices(&[1, 2], 1.5, &converter),
            RichSimplex::from_vertices(&[0, 1], 1., &converter),
        ];
        (faces, cofaces, converter)
    }

    #[test]
    fn test_update_transformation() {
        let (faces, cofaces, converter) = get_faces_and_cofaces();
        let coface_indices: HashMap<CNS, usize> = cofaces.iter().enumerate().map(|(i, c)| (*&c.simplex, i)).collect();
        let mut relevant_cofaces: HashSet<usize> = [0, 2].iter().copied().collect();
        let mut t = TransformationMatrix::new(3);
        update_transformation(&mut relevant_cofaces, &faces, &coface_indices, &converter,
                             &mut t, 0, 1);
        let expected_column: HashSet<usize> = [0, 1].iter().copied().collect();
        println!("{:?}", t.get_column(0));
        assert!(t.get_column(0).is_subset(&expected_column));
        assert!(t.get_column(0).is_superset(&expected_column));

        let expected_cofaces: HashSet<usize> = [0, 1].iter().copied().collect();
        assert!(relevant_cofaces.is_subset(&expected_cofaces));
        assert!(relevant_cofaces.is_superset(&expected_cofaces));
    }


    #[test]
    fn test_find_persistent_pairs() {
        let (faces, cofaces, converter) = get_faces_and_cofaces();

        let result = find_persistent_pairs(&faces, &cofaces, &converter);
        let expected_pairs = [(0, 1), (1, 2)];
        let expected_pairs: Vec<PersistencePair> = expected_pairs.iter().map(|(f, c)| {
                    PersistencePair{face_index: *f, coface_index: *c}
                })
                .collect();
        let expected_essentials = [2];
        assert_eq!(result.pairs, expected_pairs);
        assert_eq!(result.essentials, expected_essentials);
    }

    #[test]
    fn test_filter_faces() {
        let (faces, cofaces, converter) = get_faces_and_cofaces();

        let pairs = [(0, 1), (1, 2)];
        let mut t = TransformationMatrix::new(3);
        //t.columns[&2] = [0, 1, 2].iter().copied().collect();
    }

    #[test]
    fn test_ripser() {
        let converter = SimplexConverter::construct_for_vertex_count_and_dim(3, 3);

        let complex = [
            RichSimplex::from_vertices(&[0, 1, 2], 2., &converter),
            RichSimplex::from_vertices(&[0, 2], 2., &converter),
            RichSimplex::from_vertices(&[1, 2], 1.5, &converter),
            RichSimplex::from_vertices(&[0, 1], 0., &converter),
            RichSimplex::from_vertices(&[0], 0., &converter),
            RichSimplex::from_vertices(&[1], 0., &converter),
            RichSimplex::from_vertices(&[2], 0., &converter),
        ];

        let result = compute_barcodes(&complex, &converter);
        println!("{:?}", result);
        assert!(false);
    }
}
