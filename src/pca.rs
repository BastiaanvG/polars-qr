//! Principal components from a thin SVD.

use faer::{Mat, MatRef};
use polars::prelude::*;

use crate::least_squares::singular_value_threshold;

/// How the components are extracted.
pub struct Options {
    /// How many components to keep. `None` keeps every one the data supports.
    pub n_components: Option<usize>,
    /// Whether to subtract the column means first.
    pub centre: bool,
    /// Whether to divide the columns through by their standard deviations first.
    pub scale: bool,
}

/// The components of a matrix, with the transformation they were found under.
pub struct Pca {
    /// The column means that were subtracted, or zeros when centring was off.
    pub means: Vec<f64>,
    /// The scales the columns were divided by, or ones when scaling was off.
    pub scales: Vec<f64>,
    /// One row per component, one column per feature.
    pub components: Mat<f64>,
    /// The singular values of the transformed matrix, one per retained component.
    pub singular_values: Vec<f64>,
    /// The variance along each retained component.
    pub explained_variance: Vec<f64>,
    /// The share of the total variance each retained component carries.
    pub explained_variance_ratio: Vec<f64>,
    /// The numerical rank of the transformed matrix.
    pub rank: usize,
    /// The number of observations the components were found from.
    pub n_observations: usize,
}

/// Find the principal components of the columns of `x`.
pub fn pca(x: MatRef<'_, f64>, options: &Options) -> PolarsResult<Pca> {
    let (n, p) = (x.nrows(), x.ncols());
    if n < 2 {
        polars_bail!(ComputeError: "principal components need at least two observations");
    }

    let (transformed, means, scales) = transform(x, options);
    let svd = transformed
        .thin_svd()
        .map_err(|error| polars_err!(ComputeError: "the SVD did not converge: {:?}", error))?;

    let spectrum = svd.S().column_vector();
    let available = spectrum.nrows();
    let n_components = options.n_components.unwrap_or(available);
    if n_components == 0 {
        polars_bail!(InvalidOperation: "at least one component is required");
    }
    if n_components > available {
        polars_bail!(
            ComputeError:
            "{} components were asked for, but {} observations of {} features support at \
             most {}",
            n_components, n, p, available,
        );
    }

    let all_values: Vec<f64> = (0..available).map(|i| spectrum[i]).collect();
    let threshold = singular_value_threshold(n, p, all_values[0]);
    let rank = all_values
        .iter()
        .filter(|value| **value > threshold)
        .count();

    // The right singular vectors are the loadings; one component per row reads the same
    // way round as the rest of the results in this crate.
    let mut components = svd.V().subcols(0, n_components).transpose().to_owned();
    fix_signs(&mut components);

    // A component carries the variance of the data projected onto it, which is its
    // singular value squared over the same degrees of freedom the covariance uses. The
    // ratio is taken against every direction the data spans, not only the retained ones,
    // so dropping components does not inflate the shares of the ones that are kept.
    let degrees_of_freedom = n as f64 - 1.0;
    let variances: Vec<f64> = all_values
        .iter()
        .map(|value| value * value / degrees_of_freedom)
        .collect();
    let total: f64 = variances.iter().sum();
    let explained_variance = variances[..n_components].to_vec();
    let explained_variance_ratio = explained_variance
        .iter()
        .map(|variance| if total > 0.0 { variance / total } else { 0.0 })
        .collect();

    Ok(Pca {
        means,
        scales,
        components,
        singular_values: all_values[..n_components].to_vec(),
        explained_variance,
        explained_variance_ratio,
        rank,
        n_observations: n,
    })
}

/// Give every component a sign that does not depend on the factorisation.
///
/// A singular vector and its negation describe the same direction, and which one comes out
/// of a decomposition is an implementation detail that can differ between machines, library
/// versions or even runs. Flipping each component so that its largest entry is positive
/// pins that down, which is what makes two runs over the same data comparable. The first
/// largest entry wins when two are equally large.
fn fix_signs(components: &mut Mat<f64>) {
    for i in 0..components.nrows() {
        let mut largest = 0.0;
        let mut sign = 1.0;
        for j in 0..components.ncols() {
            let value = components[(i, j)];
            if value.abs() > largest {
                largest = value.abs();
                sign = if value < 0.0 { -1.0 } else { 1.0 };
            }
        }
        if sign < 0.0 {
            for j in 0..components.ncols() {
                components[(i, j)] = -components[(i, j)];
            }
        }
    }
}

/// Centre and scale the columns, reporting what was applied.
///
/// A column that does not vary has no scale to divide by. Dividing it by its own standard
/// deviation would be a division by zero, so it is left alone and its reported scale is one.
fn transform(x: MatRef<'_, f64>, options: &Options) -> (Mat<f64>, Vec<f64>, Vec<f64>) {
    let (n, p) = (x.nrows(), x.ncols());

    let means: Vec<f64> = if options.centre {
        (0..p)
            .map(|j| (0..n).map(|i| x[(i, j)]).sum::<f64>() / n as f64)
            .collect()
    } else {
        vec![0.0; p]
    };

    let scales: Vec<f64> = if options.scale {
        (0..p)
            .map(|j| {
                let deviation =
                    (0..n).map(|i| (x[(i, j)] - means[j]).powi(2)).sum::<f64>() / (n as f64 - 1.0);
                let scale = deviation.sqrt();
                if scale > 0.0 {
                    scale
                } else {
                    1.0
                }
            })
            .collect()
    } else {
        vec![1.0; p]
    };

    let transformed = Mat::from_fn(n, p, |i, j| (x[(i, j)] - means[j]) / scales[j]);
    (transformed, means, scales)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(rows: &[&[f64]]) -> Mat<f64> {
        Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
    }

    fn centred() -> Options {
        Options {
            n_components: None,
            centre: true,
            scale: false,
        }
    }

    #[test]
    fn finds_the_direction_the_data_varies_along() {
        // The cloud lies on the line b = 2a, so the first component points along it.
        let x = matrix(&[&[-2.0, -4.0], &[-1.0, -2.0], &[1.0, 2.0], &[2.0, 4.0]]);

        let result = pca(x.as_ref(), &centred()).unwrap();

        let direction = 1.0 / 5.0f64.sqrt();
        assert!((result.components[(0, 0)].abs() - direction).abs() < 1e-12);
        assert!((result.components[(0, 1)].abs() - 2.0 * direction).abs() < 1e-12);
        assert_eq!(result.rank, 1);
        assert_eq!(result.n_observations, 4);
    }

    #[test]
    fn components_are_orthonormal() {
        let x = matrix(&[
            &[1.0, 0.5, -1.0],
            &[2.0, -1.0, 0.0],
            &[3.0, 2.0, 1.0],
            &[-1.0, 0.0, 2.0],
        ]);

        let result = pca(x.as_ref(), &centred()).unwrap();

        for i in 0..result.components.nrows() {
            let norm: f64 = (0..3).map(|j| result.components[(i, j)].powi(2)).sum();
            assert!((norm - 1.0).abs() < 1e-12);
            for other in 0..i {
                let dot: f64 = (0..3)
                    .map(|j| result.components[(i, j)] * result.components[(other, j)])
                    .sum();
                assert!(dot.abs() < 1e-12);
            }
        }
    }

    #[test]
    fn every_component_leads_with_a_positive_entry() {
        let x = matrix(&[
            &[1.0, 0.5, -1.0],
            &[2.0, -1.0, 0.0],
            &[3.0, 2.0, 1.0],
            &[-1.0, 0.0, 2.0],
        ]);

        let result = pca(x.as_ref(), &centred()).unwrap();

        for i in 0..result.components.nrows() {
            let largest = (0..3)
                .max_by(|a, b| {
                    result.components[(i, *a)]
                        .abs()
                        .total_cmp(&result.components[(i, *b)].abs())
                })
                .unwrap();
            assert!(result.components[(i, largest)] > 0.0);
        }
    }

    #[test]
    fn negating_the_data_leaves_the_components_alone() {
        // Negating every observation negates the singular vectors, so a decomposition that
        // did not pin the signs down would return the components flipped.
        let x = matrix(&[&[1.0, 0.5], &[2.0, -1.0], &[3.0, 2.0], &[-1.0, 0.0]]);
        let negated = Mat::from_fn(4, 2, |i, j| -x[(i, j)]);

        let one = pca(x.as_ref(), &centred()).unwrap();
        let other = pca(negated.as_ref(), &centred()).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!((one.components[(i, j)] - other.components[(i, j)]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn centring_reports_the_means_it_removed() {
        let x = matrix(&[&[1.0, 10.0], &[3.0, 20.0], &[5.0, 30.0]]);

        let result = pca(x.as_ref(), &centred()).unwrap();

        assert!((result.means[0] - 3.0).abs() < 1e-12);
        assert!((result.means[1] - 20.0).abs() < 1e-12);
        assert_eq!(result.scales, [1.0, 1.0]);
    }

    #[test]
    fn scaling_puts_the_columns_on_one_footing() {
        // The second column varies a hundred times as much as the first. Without scaling
        // the first component follows it; with scaling both columns count equally.
        let x = matrix(&[&[1.0, 100.0], &[2.0, -100.0], &[3.0, 100.0], &[4.0, -100.0]]);

        let unscaled = pca(x.as_ref(), &centred()).unwrap();
        let scaled = pca(
            x.as_ref(),
            &Options {
                scale: true,
                ..centred()
            },
        )
        .unwrap();

        assert!(unscaled.components[(0, 1)].abs() > 0.99);
        assert!(scaled.components[(0, 0)].abs() > 0.1);
        assert!((scaled.scales[1] - 115.470053837925).abs() < 1e-9);
    }

    #[test]
    fn the_explained_variance_adds_up_to_the_total_variance() {
        let x = matrix(&[
            &[1.0, 0.5, -1.0],
            &[2.0, -1.0, 0.0],
            &[3.0, 2.0, 1.0],
            &[-1.0, 0.0, 2.0],
        ]);

        let result = pca(x.as_ref(), &centred()).unwrap();

        // The total variance of the components is the total variance of the columns.
        let column_variance: f64 = (0..3)
            .map(|j| {
                let mean = (0..4).map(|i| x[(i, j)]).sum::<f64>() / 4.0;
                (0..4).map(|i| (x[(i, j)] - mean).powi(2)).sum::<f64>() / 3.0
            })
            .sum();
        let explained: f64 = result.explained_variance.iter().sum();
        let ratios: f64 = result.explained_variance_ratio.iter().sum();

        assert!((explained - column_variance).abs() < 1e-12);
        assert!((ratios - 1.0).abs() < 1e-12);
    }

    #[test]
    fn the_ratios_are_taken_against_every_direction_not_only_the_kept_ones() {
        let x = matrix(&[
            &[1.0, 0.5, -1.0],
            &[2.0, -1.0, 0.0],
            &[3.0, 2.0, 1.0],
            &[-1.0, 0.0, 2.0],
        ]);

        let all = pca(x.as_ref(), &centred()).unwrap();
        let first = pca(
            x.as_ref(),
            &Options {
                n_components: Some(1),
                ..centred()
            },
        )
        .unwrap();

        assert!(
            (all.explained_variance_ratio[0] - first.explained_variance_ratio[0]).abs() < 1e-12
        );
        assert!(first.explained_variance_ratio[0] < 1.0);
    }

    #[test]
    fn the_variance_decreases_from_one_component_to_the_next() {
        let x = matrix(&[
            &[1.0, 0.5, -1.0],
            &[2.0, -1.0, 0.0],
            &[3.0, 2.0, 1.0],
            &[-1.0, 0.0, 2.0],
        ]);

        let result = pca(x.as_ref(), &centred()).unwrap();

        for pair in result.explained_variance.windows(2) {
            assert!(pair[0] >= pair[1]);
        }
    }

    #[test]
    fn keeps_only_the_components_that_were_asked_for() {
        let x = matrix(&[
            &[1.0, 0.5, -1.0],
            &[2.0, -1.0, 0.0],
            &[3.0, 2.0, 1.0],
            &[-1.0, 0.0, 2.0],
        ]);

        let result = pca(
            x.as_ref(),
            &Options {
                n_components: Some(2),
                ..centred()
            },
        )
        .unwrap();

        assert_eq!(result.components.nrows(), 2);
        assert_eq!(result.components.ncols(), 3);
        assert_eq!(result.singular_values.len(), 2);
    }

    #[test]
    fn rejects_more_components_than_the_data_supports() {
        let x = matrix(&[&[1.0, 2.0], &[3.0, 4.0]]);

        assert!(pca(
            x.as_ref(),
            &Options {
                n_components: Some(3),
                ..centred()
            }
        )
        .is_err());
    }

    #[test]
    fn rejects_a_sample_of_one() {
        let x = matrix(&[&[1.0, 2.0]]);

        assert!(pca(x.as_ref(), &centred()).is_err());
    }
}
