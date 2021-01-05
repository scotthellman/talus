fn choose(n: usize, k: usize) -> usize {
    if k == 0 {
        return 1;
    }
    // this could be tail recursive but tbh i'm going to memoize it anyway so whatever
    if k > n-k {
        return n * choose(n, n-k-1) / (n-k)
    }
    n * choose(n, k-1) / k
}

fn binary_search_for_cns(n: CNS, d: Dimension, limit: usize) -> CNS {
    let d = usize::from(d);
    let n = usize::from(n);

    let mut upper = limit;
    let mut lower = 0;
    let mut mid;

    while upper > lower {
        mid = (upper + lower) / 2;
        let value = choose(mid, d+1);

        if value <= n {
            lower = mid + 1
        } else {
            upper = mid - 1
        }
    }
    if choose(lower, d+1) > n {  
        lower -= 1
    }
    CNS::from(lower)
}


#[derive(Clone, Copy)]
struct CNS(usize);
impl From<usize> for CNS {
    fn from(val: usize) -> CNS {
        CNS(val)
    }
}

impl From<CNS> for usize {
   fn from(n: CNS) -> usize {
       n.0
   }
}

struct Dimension(usize);
impl From<usize> for Dimension {
    fn from(val: usize) -> Dimension {
        Dimension(val)
    }
}

impl From<Dimension> for usize {
   fn from(d: Dimension) -> usize {
       d.0
   }
}


struct Simplex {
    // TODO: this is related to the LabeledPoint
    vertices: Vec<usize>,
    value: f64,
}


impl Simplex {
    fn construct_simplex(vertices: &[usize], value: f64) -> Self {
        let mut sorted_vertices: Vec<usize> = vertices.iter()
            .cloned()
            .collect();
        sorted_vertices.sort();
        Simplex{vertices: sorted_vertices, value}
    }

    // FIXME: this can be done as a proper conversion (other way can't be)
    fn to_cns(&self) -> CNS {
        let value: usize = self.vertices.iter().enumerate()
            .map(|(i, &v)| choose(i, v))
            .sum();
        CNS::from(value)
    }
}


impl CNS {
    fn to_vector(self, dim: Dimension) -> Vec<usize> {
        let dim = usize::from(dim);
        let mut vertices = Vec::<usize>::with_capacity(dim);
        let mut n = usize::from(self);
        let mut limit = if n > dim {n} else {dim};

        for d in (0..dim+1).rev() {
            limit = usize::from(binary_search_for_cns(self, Dimension::from(d), limit));
            vertices.push(limit);
            n -= choose(limit, d+1)
        }
        vertices
    }

    fn to_simplex(self, dim: Dimension, value: f64) -> Simplex {
        Simplex {
            vertices: self.to_vector(dim),
            value
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cns_isomorphism() {
        let simplex = Simplex::construct_simplex(&vec![5, 3, 1], 3.0);
        assert_eq!(simplex.vertices, simplex.to_cns().to_vector(Dimension::from(3)))
    }
}
