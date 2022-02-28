/// Boolean operations on BezPaths
use crate::BezPath;

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
       const KEEP_A_OUTSIDE_B   = 0b00000101;
       const KEEP_A_INSIDE_B    = 0b00000110;
       const KEEP_B_INSIDE_A    = 0b00001001;
       const KEEP_B_OUTSIDE_A   = 0b00001010;
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
    use crate::{Circle, Point, Shape};

    #[test]
    fn test_circle_unite_circle() {
        let c1 = Circle::new(Point::new(0., 0.), 10.).to_path(1e-3);
        let c2 = Circle::new(Point::new(5., 0.), 10.).to_path(1e-3);
        let res = path_path_operation(&c1, PathOperation::Subtract(&c2));
        // @TODO DO THIS
        //assert_eq!(res.elements().len(), 8);
    }
}