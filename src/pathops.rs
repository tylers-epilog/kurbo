/// This code in this file peformes boolean operations on BezPaths
use crate::curve_intersections::*;
use crate::{BezPath, ParamCurve, ParamCurveExtrema, PathSeg, Point};
use arrayvec::ArrayVec;

/// An operation that can be performed on a path
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PathOperation<'a> {
    /// Subtracts the other path from the current path
    Subtract(&'a BezPath),
    /// Like subtract but doesn't close current path along second path
    Outside(&'a BezPath),
    /// Gets the inersection of this path with another
    Intersect(&'a BezPath),
    /// Like intersection but doesn't close current path along second path
    Inside(&'a BezPath),
    /// Unites this path with another
    UniteWith(&'a BezPath),
    /// Gets the exclusive-or operation with another path
    Xor(&'a BezPath),
}

bitflags::bitflags! {
    // private struct, used in do_path_op
    struct ClosedPathOperationsFlags: u32 {
       const DISCARD_SEGS       = 0b00000000;
       const KEEP_A_OUTSIDE_B   = 0b00000001;
       const KEEP_A_INSIDE_B    = 0b00000010;
       const KEEP_B_INSIDE_A    = 0b00000100;
       const KEEP_B_OUTSIDE_A   = 0b00001000;
       const KEEP_ALL_SEGS      = Self::KEEP_A_OUTSIDE_B.bits
                                | Self::KEEP_A_INSIDE_B.bits
                                | Self::KEEP_B_INSIDE_A.bits
                                | Self::KEEP_B_OUTSIDE_A.bits;
    }
}

/// Compute an operation between two paths
pub fn path_path_operation(path_a: &BezPath, operation: PathOperation) -> BezPath {
    match operation {
        PathOperation::Subtract(path_b) => {
            // Subtract b from a
            do_path_op(
                path_a,
                path_b,
                ClosedPathOperationsFlags::KEEP_A_OUTSIDE_B
                    | ClosedPathOperationsFlags::KEEP_B_INSIDE_A,
                true,
            )
        }
        PathOperation::Outside(path_b) => {
            // Get a outside of b
            do_path_op(
                path_a,
                path_b,
                ClosedPathOperationsFlags::KEEP_A_OUTSIDE_B,
                false,
            )
        }
        PathOperation::Intersect(path_b) => {
            // Get the intersectionf of a and b
            do_path_op(
                path_a,
                path_b,
                ClosedPathOperationsFlags::KEEP_A_INSIDE_B
                    | ClosedPathOperationsFlags::KEEP_B_INSIDE_A,
                true,
            )
        }
        PathOperation::Inside(path_b) => {
            // Get a inside of b
            do_path_op(
                path_a,
                path_b,
                ClosedPathOperationsFlags::KEEP_A_INSIDE_B,
                false,
            )
        }
        PathOperation::UniteWith(path_b) => {
            // Units a and b
            do_path_op(
                path_a,
                path_b,
                ClosedPathOperationsFlags::KEEP_A_OUTSIDE_B
                    | ClosedPathOperationsFlags::KEEP_B_OUTSIDE_A,
                true,
            )
        }
        PathOperation::Xor(path_b) => {
            // Get a XOR b
            // We can actually do a simple shortcut with this one. All we have
            // to do is append path_b to path_a. Since they are both even-odd
            // fill, everything is taken care of just by the logic of the fill.
            let mut path_final = path_a.clone();
            path_final.extend(path_b.iter());
            path_final
        }
    }
}

/// Gets possible self intersections for a given path segment, ignoring the end
/// points
fn segment_self_intersections(segment: &PathSeg, accuracy: f64) -> ArrayVec<(f64, f64), 9> {
    match segment {
        PathSeg::Cubic(bez) => {
            if Point::is_near(bez.p0, bez.p3, accuracy) {
                // Only the endpoints intersect, which we will ignore
                return ArrayVec::<(f64, f64), 9>::new();
            }

            // I'll make the argument (based on experiemntation) that if we
            // take any cubic bezier where the start and end points are
            // horizontal to each other, and split it at the maximum y
            // posiiton, the two halves will intersect IFF the cibic bezier
            // curve is self-intersecting.

            // The first step in finding this split point is to transform the
            // bezier curve so that start/end points are horizontal.
            let bez_uv = bez.transform_to_uv();

            // Then find the t-value of the largest peak
            let extrema_y = bez_uv.extrema_y();
            let max_peak = extrema_y.iter().fold(0., |acc, t| {
                let pt_acc = bez_uv.eval(acc);
                let pt_t = bez_uv.eval(*t);
                if pt_t.y.abs() > pt_acc.y.abs() {
                    *t
                } else {
                    acc
                }
            });

            // Split the original bezier at the location of the peak
            let (domain1, domain2) = &(0.0..max_peak, max_peak..1.0);
            let (curve1, curve2) = (
                bez.subsegment(domain1.clone()),
                bez.subsegment(domain2.clone()),
            );

            // Return the intersections of the two curves, not including end-points
            curve_curve_intersections(&curve1, &curve2, CurveIntersectionFlags::NONE, accuracy)
                .into_iter()
                .map(|(t1, t2)| {
                    (
                        domain_value_at_t(domain1, t1),
                        domain_value_at_t(domain2, t2),
                    )
                })
                .collect::<ArrayVec<_, 9>>()
        }
        // No other segment type can self intersect
        _ => ArrayVec::<(f64, f64), 9>::new(),
    }
}

/// Gets the index and t-parameter of each segment in the path that intersects
/// with another segment in the path
fn path_self_intersections(path: &BezPath, accuracy: f64) -> Vec<((usize, f64), (usize, f64))> {
    let segments = path
        .iter()
        .enumerate()
        .filter_map(|(index, _)| match path.get_seg(index) {
            Some(seg) => Some((index, seg)),
            None => None,
        })
        .collect::<Vec<_>>();

    // Find self-intersecting segments
    segments
        .iter()
        .by_ref()
        .map(|&(index, seg)| {
            segment_self_intersections(&seg, accuracy)
                .into_iter() // Get self-intersections for each segment
                .map(|intersect| ((index, intersect.0), (index, intersect.1)))
                .chain(
                    segments
                        .iter()
                        .filter(|&(index2, _)| *index2 > index)
                        .map(|&(index2, seg2)| {
                            let flags = if index == 1 && index2 == path.elements().len() - 1 {
                                // Don't capture the endpoint intersection caused by the paths being closed
                                CurveIntersectionFlags::KEEP_CURVE1_T1_INTERSECTION
                                    | CurveIntersectionFlags::KEEP_CURVE2_T0_INTERSECTION
                            } else if index2 == index + 1 {
                                // Don't capture the endpoint intersection caused by the segments being sequential
                                CurveIntersectionFlags::KEEP_CURVE1_T0_INTERSECTION
                                    | CurveIntersectionFlags::KEEP_CURVE2_T1_INTERSECTION
                            } else {
                                // Capture all endpoint intersections
                                CurveIntersectionFlags::KEEP_ALL_ENDPOINT_INTERSECTIONS
                            };
                            curve_curve_intersections(&seg, &seg2, flags, accuracy)
                                .into_iter() // Get intersections with segments after this one
                                .map(|intersect| ((index, intersect.0), (index2, intersect.1)))
                                .collect::<Vec<_>>()
                        })
                        .flatten(),
                )
                .collect::<Vec<_>>()
        })
        .flatten()
        .collect::<Vec<_>>()
}

/// This function converts a winding-fill path into an even-odd filled path
/// Note: Since there is no winding fill property of path, the path is assumed
/// to have a winding fill when this function is called
/// Note: Open paths are closed
pub fn convert_path_to_even_odd(path: &BezPath) -> BezPath {
    // @TODO DO THIS
    // @TODO CLOSE OPEN PATHS
    path.clone() // TEMPORARY CODE
}

/// Performs an operation on the current path
/// Note: Paths are assumed to be even-odd filled
fn do_path_op(
    path_a: &BezPath,
    path_b: &BezPath,
    closed_flags: ClosedPathOperationsFlags,
    connect_nearby_paths: bool,
) -> BezPath {
    // @TODO DO THIS
    // @TODO CLOSE OPEN PATHS
    path_a.clone() // TEMPORARY CODE
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Circle, Shape};

    #[test]
    fn test_circle_unite_circle() {
        let c1 = Circle::new((0., 0.), 10.).to_path(1e-3);
        let c2 = Circle::new((5., 0.), 10.).to_path(1e-3);
        let res = path_path_operation(&c1, PathOperation::Subtract(&c2));
        // @TODO DO THIS
        //assert_eq!(res.elements().len(), 8);
    }

    #[test]
    fn test_self_intersections() {
        let mut path = BezPath::new();
        path.move_to((0., 0.));
        path.line_to((0., 1.));
        path.curve_to((2., 0.5), (-1., 0.5), (1., 1.));
        path.line_to((-0.5, 0.));
        path.close_path();
        let mut path2 = path.clone();
        let intersects = path_self_intersections(&path, 0.);
        path2.break_at_intersections(&intersects);
        assert_eq!(intersects.len(), 4);
        assert_eq!(
            path2.elements().len(),
            path.elements().len() + intersects.len() * 4,
        );
    }
}
