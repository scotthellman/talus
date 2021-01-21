struct BinomialCoeff {
    coeffs:  Vec<Vec<usize>>
}

impl BinomialCoeff {
    fn construct_for_max_k_and_n(n: usize, k: usize) -> Self {
        let mut coeffs: Vec<Vec<usize>> = Vec::with_capacity(n);
        for i in 0..n+1 {
            let row = (0..k+1).map(|j| {
                if j > i {
                    0
                }
                else if j == 0 {
                    1
                }
                else {
                    coeffs[i-1][j-1] + coeffs[i-1][j]
                }
            }).collect();
            coeffs.push(row);
        }
        BinomialCoeff{coeffs}
    }

    fn binomial(&self, n: usize, k: usize) -> usize {
        self.coeffs.get(n).map_or(0, |row| *row.get(k).unwrap_or(&0))
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn choose(n: usize, k: usize) -> usize {
        // Another way of calculating choose that
        // doesn't require all the intermediate numbers
        // but is very vulnerable to overflow
        if k > n {
            return 0
        };

        let mut numerator = 1;
        let mut denominator = 1;
        for i in 1..k+1 {
            numerator *= n + 1 - i;
            denominator *= i;
        }
        numerator / denominator
    }

    #[test]
    fn test_simple_binomial() {
        let coeffs = BinomialCoeff::construct_for_max_k_and_n(50, 10);
        assert_eq!(coeffs.binomial(1,1), 1);
        assert_eq!(coeffs.binomial(2,1), 2);
        assert_eq!(coeffs.binomial(2,2), 1);
        assert_eq!(coeffs.binomial(4,2), 6);
        assert_eq!(coeffs.binomial(5,3), 10);
        assert_eq!(coeffs.binomial(15,5), 3003);
        assert_eq!(coeffs.binomial(50,10), 10272278170);
    }

    proptest! {
        #[test]
        fn test_binomial(n in 0usize..20, d in 1usize..3) {
            let coeffs = BinomialCoeff::construct_for_max_k_and_n(n, d);
            prop_assert_eq!(coeffs.binomial(n, d), choose(n, d));
        }
    }
}
