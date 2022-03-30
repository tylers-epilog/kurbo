/// Boolean operations on BezPaths
///
/// The methodology approximately follows that which is described here:
/// https://losingfight.com/blog/2011/07/07/how-to-implement-boolean-operations-on-bezier-paths-part-1/
/// https://losingfight.com/blog/2011/07/08/how-to-implement-boolean-operations-on-bezier-paths-part-2/
/// https://losingfight.com/blog/2011/07/09/how-to-implement-boolean-operations-on-bezier-paths-part-3/
use crate::{BezPath, Point, Rect, Shape};

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
            // Get the intersections of a and b
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

/// This function converts a winding-fill path into an even-odd filled path
/// Note: Since there is no winding fill property of path, the path is assumed
/// to have a winding fill when this function is called
/// Note: Open paths are closed
pub fn convert_path_to_even_odd(path: &BezPath, accuracy: f64) -> BezPath {
    // Make a copy of the path
    let mut path = path.clone();

    // Close all subpaths because this path is treated as being closed anyways
    path.close_subpaths();

    // Break apart at self intersections
    BezPath::break_at_self_intersections(&mut path, accuracy);
    let paths = path.split_at_moves();

    // Convert the list of elements into a vector of tuples to make tracking elements easier
    let mut elements = paths
        .iter()
        .enumerate()
        .map(|(index, path)| {
            // Create a tuple
            (index, path, path.bounding_box(), false, false)
        })
        .collect::<Vec<_>>();

    while let Some(first_unprocessed) = elements.iter().find(|(_, _, _, processed, _)| !*processed)
    {
        // Pick the first unprocessed element and create a ray through it to test if we should keep it
        let (index1, _path1, bb1, _processed1, _keep1) = *first_unprocessed;

        // Create a ray that travels through our path
        let ray_is_horizontal = bb1.width().abs() <= bb1.height().abs();
        let ray_position = if ray_is_horizontal {
            (bb1.min_y() + bb1.max_y()) / 2.
        } else {
            (bb1.min_x() + bb1.max_x()) / 2.
        };
        let ray_point = if ray_is_horizontal {
            Point::new(bb1.max_x() + 1., ray_position)
        } else {
            Point::new(ray_position, bb1.max_y() + 1.)
        };
        let ray_fn = if ray_is_horizontal {
            BezPath::winding_x_results
        } else {
            BezPath::winding_y_results
        };
        let bounds_fn = |bb: Rect| {
            if ray_is_horizontal {
                bb.min_y()..bb.max_y()
            } else {
                bb.min_x()..bb.max_x()
            }
        };

        // Loop through all paths and determine which ones intersect our ray
        let mut winding_results = elements
            .iter()
            .by_ref()
            .filter(|(index2, _, bb2, _, _)| {
                index1 == *index2 || bounds_fn(*bb2).contains(&ray_position)
            })
            .map(|(index2, path2, _, processed2, _)| {
                let winding_array = ray_fn(&path2, ray_point)
                    .into_iter()
                    .map(|(pt, winding)| (pt, winding, index2.clone(), processed2.clone()))
                    .collect::<Vec<_>>();
                winding_array
            })
            .flatten()
            .collect::<Vec<_>>();

        // Sort by the position of the intersection with the ray
        winding_results.sort_by(|(a, _, _, _), (b, _, _, _)| (a).partial_cmp(b).unwrap());

        winding_results
            .into_iter()
            .scan(0, |winding_total, (p, w, index2, processed2)| {
                let winding_before = *winding_total;
                *winding_total = *winding_total + w;
                Some((p, winding_before, *winding_total, index2, processed2))
            })
            .for_each(|(_, w_before, w_current, index2, _)| {
                let (_, _, _, ref mut processed2_ref, ref mut keep2_ref) = elements[index2];

                // Discard if crossing the path didn't change whether or not we're inside the original path
                *keep2_ref = (w_before != 0) != (w_current != 0);

                // Mark as processed
                *processed2_ref = true;
            });
    }

    // Remove flagged sections of the path
    let mut elements = elements
        .into_iter()
        .filter_map(|(_, path, _, _, keep)| if keep { Some((path, false)) } else { None })
        .filter(|(path, _)| !path.elements().is_empty())
        .collect::<Vec<_>>();

    // Re-close paths
    for index1 in 0..elements.len() {
        if let Some(path_end1) = elements[index1]
            .0
            .get_seg_end(elements[index1].0.elements().len() - 1)
        {
            if let Some((connected_index, _)) = elements
                .iter()
                .enumerate()
                .filter(|(index2, _)| *index2 > index1)
                .find(|(_, (path2, _))| {
                    if let Some(path_start2) = path2.get_seg_end(0) {
                        return Point::is_near(path_end1, path_start2, accuracy);
                    }
                    false
                })
            {
                // Set a flag so that we connect to path1 later by removing path2's move-to
                elements[connected_index].1 = true;

                // Move index to be next in line
                elements.swap(index1 + 1, connected_index);
            }
        }
    }
    if !elements.is_empty() {
        assert!(!elements[0].1); // Fist element can't conenct to the previous one
    }
    let elements = elements
        .into_iter()
        .map(|(path, connect_to_previous)| {
            BezPath::from_vec(path.elements()[if connect_to_previous { 1 } else { 0 }..].to_vec())
        })
        .flatten()
        .collect::<Vec<_>>();

    BezPath::from_vec(elements)
}

/// Performs an operation on the current path
/// Note: Paths are assumed to be even-odd filled
fn do_path_op(
    path_a: &BezPath,
    path_b: &BezPath,
    closed_flags: ClosedPathOperationsFlags,
    connect_nearby_paths: bool,
) -> BezPath {
    // Make copies of the paths
    let mut path_a = path_a.clone();
    let mut path_b = path_b.clone();

    if connect_nearby_paths {
        // We treat all paths as if they are closed, so we'll just go ahead and
        // close the paths
        path_a.close_subpaths();
        path_b.close_subpaths();
    }

    // Find all intersections between the two paths
    // let intersections = path_path_intersections(&path_a, &path_b);

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
    fn test_convert_to_even_odd() {
        let mut path = BezPath::new();
        path.move_to((0., 0.));
        path.line_to((0., 1.));
        path.curve_to((2., 0.5), (-1., 0.5), (1., 1.));
        path.line_to((-0.5, 0.));
        path.close_path();

        let path2 = convert_path_to_even_odd(&path, 0.);

        assert_eq!(path2.elements().len(), 10);
    }
}
