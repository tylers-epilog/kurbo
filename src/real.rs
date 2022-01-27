//! A utility for comparing real values

use crate::{Point};

/// If we're comparing numbers, our epsilon should depend on how big the number
/// is. This function returns an epsilon appropriate for the size the largest
/// number.
pub fn epsilon_for_value(num: f64) -> f64 {
    // These values have been obtained experimentally and can be changed if
    // necessary.
    // (num.abs() * 0.0000000001).max(0.0000001)
    (num.abs() * 0.0000000001).max(f64::EPSILON)
}

/// If we're comparing distances between samples of curves, our epsilon should
/// depend on how big the points we're comparing are. This function returns an
/// epsilon appropriate for the size of pt.
pub fn epsilon_for_point(pt: Point) -> f64 {
    let max = f64::max(f64::abs(pt.x), f64::abs(pt.y));
    epsilon_for_value(max)
}

/// Compare if two real values are approximately equal
pub fn real_is_equal(num1: f64, num2: f64) -> bool
{
    (num1 - num2).abs() < epsilon_for_value(num1.max(num2))
}

/// Compare if a  real value is approximately zero
pub fn real_is_zero(num1: f64) -> bool
{
    real_is_equal(num1, 0.)
}

/// Compare if num1 < num2 and not approximately equal
pub fn real_lt(num1: f64, num2: f64) -> bool
{
    (num1 < num2) && !real_is_equal(num1, num2)
}

/// Compare if num1 < num2 or approximately equal
pub fn real_lte(num1: f64, num2: f64) -> bool
{
    (num1 < num2) || real_is_equal(num1, num2)
}

/// Compare if num1 > num2 and not approximately equal
pub fn real_gt(num1: f64, num2: f64) -> bool
{
    (num1 > num2) && !real_is_equal(num1, num2)
}

/// Compare if num1 > num2 or approximately equal
pub fn real_gte(num1: f64, num2: f64) -> bool
{
    (num1 > num2) || real_is_equal(num1, num2)
}

/// Compare if two points are approximately equal
pub fn point_is_equal(pt1: Point, pt2: Point) -> bool
{
    real_is_equal(pt1.x, pt2.x) && real_is_equal(pt1.y, pt2.y)
}
