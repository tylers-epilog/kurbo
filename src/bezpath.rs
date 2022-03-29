//! Bézier paths (up to cubic).

#![allow(clippy::many_single_char_names)]

use std::iter::{Extend, FromIterator};
use std::mem;
use std::ops::{Mul, Range};

use arrayvec::ArrayVec;

use crate::common::{solve_cubic, solve_quadratic};
use crate::curve_intersections::{
    curve_curve_intersections, domain_value_at_t, CurveIntersectionFlags,
};
use crate::MAX_EXTREMA;
use crate::{
    Affine, ConstPoint, CubicBez, Line, Nearest, ParamCurve, ParamCurveArclen, ParamCurveArea,
    ParamCurveDeriv, ParamCurveExtrema, ParamCurveNearest, Point, QuadBez, Rect, Shape,
    TranslateScale, Vec2,
};

/// A Bézier path.
///
/// These docs assume basic familiarity with Bézier curves; for an introduction,
/// see Pomax's wonderful [A Primer on Bézier Curves].
///
/// This path can contain lines, quadratics ([`QuadBez`]) and cubics
/// ([`CubicBez`]), and may contain multiple subpaths.
///
/// # Elements and Segments
///
/// A Bézier path can be represented in terms of either 'elements' ([`PathEl`])
/// or 'segments' ([`PathSeg`]). Elements map closely to how Béziers are
/// generally used in PostScript-style drawing APIs; they can be thought of as
/// instructions for drawing the path. Segments more directly describe the
/// path itself, with each segment being an independent line or curve.
///
/// These different representations are useful in different contexts.
/// For tasks like drawing, elements are a natural fit, but when doing
/// hit-testing or subdividing, we need to have access to the segments.
///
/// Conceptually, a `BezPath` contains zero or more subpaths. Each subpath
/// *always* begins with a `MoveTo`, then has zero or more `LineTo`, `QuadTo`,
/// and `CurveTo` elements, and optionally ends with a `ClosePath`.
///
/// Internally, a `BezPath` is a list of [`PathEl`]s; as such it implements
/// [`FromIterator<PathEl>`] and [`Extend<PathEl>`]:
///
/// ```
/// use kurbo::{BezPath, Rect, Shape, Vec2};
/// let accuracy = 0.1;
/// let rect = Rect::from_origin_size((0., 0.,), (10., 10.));
/// // these are equivalent
/// let path1 = rect.to_path(accuracy);
/// let path2: BezPath = rect.path_elements(accuracy).collect();
///
/// // extend a path with another path:
/// let mut path = rect.to_path(accuracy);
/// let shifted_rect = rect + Vec2::new(5.0, 10.0);
/// path.extend(shifted_rect.to_path(accuracy));
/// ```
///
/// You can iterate the elements of a `BezPath` with the [`iter`] method,
/// and the segments with the [`segments`] method:
///
/// ```
/// use kurbo::{BezPath, Line, PathEl, PathSeg, Point, Rect, Shape};
/// let accuracy = 0.1;
/// let rect = Rect::from_origin_size((0., 0.,), (10., 10.));
/// // these are equivalent
/// let path = rect.to_path(accuracy);
/// let first_el = PathEl::MoveTo(Point::ZERO);
/// let first_seg = PathSeg::Line(Line::new((0., 0.), (10., 0.)));
/// assert_eq!(path.iter().next(), Some(first_el));
/// assert_eq!(path.segments().next(), Some(first_seg));
/// ```
/// In addition, if you have some other type that implements
/// `Iterator<Item=PathEl>`, you can adapt that to an iterator of segments with
/// the [`segments` free function].
///
/// # Advanced functionality
///
/// In addition to the basic API, there are several useful pieces of advanced
/// functionality available on `BezPath`:
///
/// - [`flatten`] does Bézier flattening, converting a curve to a series of
/// line segments
/// - [`intersect_line`] computes intersections of a path with a line, useful
/// for things like subdividing
///
/// [A Primer on Bézier Curves]: https://pomax.github.io/bezierinfo/
/// [`iter`]: BezPath::iter
/// [`segments`]: BezPath::segments
/// [`flatten`]: BezPath::flatten
/// [`intersect_line`]: PathSeg::intersect_line
/// [`segments` free function]: segments
/// [`FromIterator<PathEl>`]: std::iter::FromIterator
/// [`Extend<PathEl>`]: std::iter::Extend
#[derive(Clone, Default, Debug, PartialEq)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BezPath(Vec<PathEl>);

/// The element of a Bézier path.
///
/// A valid path has `MoveTo` at the beginning of each subpath.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PathEl {
    /// Move directly to the point without drawing anything, starting a new
    /// subpath.
    MoveTo(Point),
    /// Draw a line from the current location to the point.
    LineTo(Point),
    /// Draw a quadratic bezier using the current location and the two points.
    QuadTo(Point, Point),
    /// Draw a cubic bezier using the current location and the three points.
    CurveTo(Point, Point, Point),
    /// Close off the path.
    ClosePath,
}

/// A segment of a Bézier path.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PathSeg {
    /// A line segment.
    Line(Line),
    /// A quadratic bezier segment.
    Quad(QuadBez),
    /// A cubic bezier segment.
    Cubic(CubicBez),
}

/// The derivative of a segment of a Bézier path.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PathSegDeriv {
    /// A point
    Point(ConstPoint),
    /// A line segment.
    Line(Line),
    /// A quadratic bezier segment.
    Quad(QuadBez),
}

/// An intersection of a [`Line`] and a [`PathSeg`].
///
/// This can be generated with the [`PathSeg::intersect_line`] method.
#[derive(Debug, Clone, Copy)]
pub struct LineIntersection {
    /// The 'time' that the intersection occurs, on the line.
    ///
    /// This value is in the range 0..1.
    pub line_t: f64,

    /// The 'time' that the intersection occurs, on the path segment.
    ///
    /// This value is nominally in the range 0..1, although it may slightly exceed
    /// that range at the boundaries of segments.
    pub segment_t: f64,
}

/// The minimum distance between two Bézier curves.
pub struct MinDistance {
    /// The shortest distance between any two points on the two curves.
    pub distance: f64,
    /// The position of the nearest point on the first curve, as a parameter.
    ///
    /// To resolve this to a [`Point`], use [`ParamCurve::eval`].
    ///
    /// [`ParamCurve::eval`]: crate::ParamCurve::eval
    pub t1: f64,
    /// The position of the nearest point on the second curve, as a parameter.
    ///
    /// To resolve this to a [`Point`], use [`ParamCurve::eval`].
    ///
    /// [`ParamCurve::eval`]: crate::ParamCurve::eval
    pub t2: f64,
}

impl BezPath {
    /// Create a new path.
    pub fn new() -> BezPath {
        Default::default()
    }

    /// Create a path from a vector of path elements.
    ///
    /// `BezPath` also implements `FromIterator<PathEl>`, so it works with `collect`:
    ///
    /// ```
    /// // a very contrived example:
    /// use kurbo::{BezPath, PathEl};
    ///
    /// let path = BezPath::new();
    /// let as_vec: Vec<PathEl> = path.into_iter().collect();
    /// let back_to_path: BezPath = as_vec.into_iter().collect();
    /// ```
    pub fn from_vec(v: Vec<PathEl>) -> BezPath {
        BezPath(v)
    }

    /// Push a generic path element onto the path.
    pub fn push(&mut self, el: PathEl) {
        self.0.push(el)
    }

    /// Push a "move to" element onto the path.
    pub fn move_to<P: Into<Point>>(&mut self, p: P) {
        self.push(PathEl::MoveTo(p.into()));
    }

    /// Push a "line to" element onto the path.
    ///
    /// Will panic with a debug assert when the current subpath does not
    /// start with `move_to`.
    pub fn line_to<P: Into<Point>>(&mut self, p: P) {
        debug_assert!(self.is_open_subpath(), "no open subpath (missing MoveTo)");
        self.push(PathEl::LineTo(p.into()));
    }

    /// Push a "quad to" element onto the path.
    ///
    /// Will panic with a debug assert when the current subpath does not
    /// start with `move_to`.
    pub fn quad_to<P: Into<Point>>(&mut self, p1: P, p2: P) {
        debug_assert!(self.is_open_subpath(), "no open subpath (missing MoveTo)");
        self.push(PathEl::QuadTo(p1.into(), p2.into()));
    }

    /// Push a "curve to" element onto the path.
    ///
    /// Will panic with a debug assert when the current subpath does not
    /// start with `move_to`.
    pub fn curve_to<P: Into<Point>>(&mut self, p1: P, p2: P, p3: P) {
        debug_assert!(self.is_open_subpath(), "no open subpath (missing MoveTo)");
        self.push(PathEl::CurveTo(p1.into(), p2.into(), p3.into()));
    }

    /// Push a "close path" element onto the path.
    ///
    /// Will panic with a debug assert when the current subpath does not
    /// start with `move_to`.
    pub fn close_path(&mut self) {
        debug_assert!(self.is_open_subpath(), "no open subpath (missing MoveTo)");
        self.push(PathEl::ClosePath);
    }

    fn is_open_subpath(&self) -> bool {
        !self.0.is_empty() && self.0.last() != Some(&PathEl::ClosePath)
    }

    /// Get the path elements.
    pub fn elements(&self) -> &[PathEl] {
        &self.0
    }

    /// Returns an iterator over the path's elements.
    pub fn iter(&self) -> impl Iterator<Item = PathEl> + '_ {
        self.0.iter().copied()
    }

    /// Iterate over the path segments.
    pub fn segments(&self) -> impl Iterator<Item = PathSeg> + '_ {
        segments(self.iter())
    }

    /// Compute the winding number contribution and points of intersection of a single segment along a horizontal ray.
    ///
    /// Cast a ray to the left and record intersections.
    pub fn winding_x_results(&self, p: Point) -> Vec<(f64, i32)> {
        self.segments()
            .into_iter()
            .map(|seg| seg.winding_x_results(p))
            .flatten()
            .collect()
    }

    /// Compute the winding number contribution and points of intersection of a single segment along a vertical ray.
    ///
    /// Cast a ray to the upwards and record intersections.
    pub fn winding_y_results(&self, p: Point) -> Vec<(f64, i32)> {
        self.segments()
            .into_iter()
            .map(|seg| seg.winding_y_results(p))
            .flatten()
            .collect()
    }

    /// Flatten the path, invoking the callback repeatedly.
    ///
    /// Flattening is the action of approximating a curve with a succession of line segments.
    ///
    /// <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 30" height="30mm" width="120mm">
    ///   <path d="M26.7 24.94l.82-11.15M44.46 5.1L33.8 7.34" fill="none" stroke="#55d400" stroke-width=".5"/>
    ///   <path d="M26.7 24.94c.97-11.13 7.17-17.6 17.76-19.84M75.27 24.94l1.13-5.5 2.67-5.48 4-4.42L88 6.7l5.02-1.6" fill="none" stroke="#000"/>
    ///   <path d="M77.57 19.37a1.1 1.1 0 0 1-1.08 1.08 1.1 1.1 0 0 1-1.1-1.08 1.1 1.1 0 0 1 1.08-1.1 1.1 1.1 0 0 1 1.1 1.1" color="#000" fill="none" stroke="#030303" stroke-linecap="round" stroke-opacity=".5"/>
    ///   <path d="M77.57 19.37a1.1 1.1 0 0 1-1.08 1.08 1.1 1.1 0 0 1-1.1-1.08 1.1 1.1 0 0 1 1.08-1.1 1.1 1.1 0 0 1 1.1 1.1" color="#000" fill="#fff"/>
    ///   <path d="M80.22 13.93a1.1 1.1 0 0 1-1.1 1.1 1.1 1.1 0 0 1-1.08-1.1 1.1 1.1 0 0 1 1.1-1.08 1.1 1.1 0 0 1 1.08 1.08" color="#000" fill="none" stroke="#030303" stroke-linecap="round" stroke-opacity=".5"/>
    ///   <path d="M80.22 13.93a1.1 1.1 0 0 1-1.1 1.1 1.1 1.1 0 0 1-1.08-1.1 1.1 1.1 0 0 1 1.1-1.08 1.1 1.1 0 0 1 1.08 1.08" color="#000" fill="#fff"/>
    ///   <path d="M84.08 9.55a1.1 1.1 0 0 1-1.08 1.1 1.1 1.1 0 0 1-1.1-1.1 1.1 1.1 0 0 1 1.1-1.1 1.1 1.1 0 0 1 1.08 1.1" color="#000" fill="none" stroke="#030303" stroke-linecap="round" stroke-opacity=".5"/>
    ///   <path d="M84.08 9.55a1.1 1.1 0 0 1-1.08 1.1 1.1 1.1 0 0 1-1.1-1.1 1.1 1.1 0 0 1 1.1-1.1 1.1 1.1 0 0 1 1.08 1.1" color="#000" fill="#fff"/>
    ///   <path d="M89.1 6.66a1.1 1.1 0 0 1-1.08 1.1 1.1 1.1 0 0 1-1.08-1.1 1.1 1.1 0 0 1 1.08-1.08 1.1 1.1 0 0 1 1.1 1.08" color="#000" fill="none" stroke="#030303" stroke-linecap="round" stroke-opacity=".5"/>
    ///   <path d="M89.1 6.66a1.1 1.1 0 0 1-1.08 1.1 1.1 1.1 0 0 1-1.08-1.1 1.1 1.1 0 0 1 1.08-1.08 1.1 1.1 0 0 1 1.1 1.08" color="#000" fill="#fff"/>
    ///   <path d="M94.4 5a1.1 1.1 0 0 1-1.1 1.1A1.1 1.1 0 0 1 92.23 5a1.1 1.1 0 0 1 1.08-1.08A1.1 1.1 0 0 1 94.4 5" color="#000" fill="none" stroke="#030303" stroke-linecap="round" stroke-opacity=".5"/>
    ///   <path d="M94.4 5a1.1 1.1 0 0 1-1.1 1.1A1.1 1.1 0 0 1 92.23 5a1.1 1.1 0 0 1 1.08-1.08A1.1 1.1 0 0 1 94.4 5" color="#000" fill="#fff"/>
    ///   <path d="M76.44 25.13a1.1 1.1 0 0 1-1.1 1.1 1.1 1.1 0 0 1-1.08-1.1 1.1 1.1 0 0 1 1.1-1.1 1.1 1.1 0 0 1 1.08 1.1" color="#000" fill="none" stroke="#030303" stroke-linecap="round" stroke-opacity=".5"/>
    ///   <path d="M76.44 25.13a1.1 1.1 0 0 1-1.1 1.1 1.1 1.1 0 0 1-1.08-1.1 1.1 1.1 0 0 1 1.1-1.1 1.1 1.1 0 0 1 1.08 1.1" color="#000" fill="#fff"/>
    ///   <path d="M27.78 24.9a1.1 1.1 0 0 1-1.08 1.08 1.1 1.1 0 0 1-1.1-1.08 1.1 1.1 0 0 1 1.1-1.1 1.1 1.1 0 0 1 1.08 1.1" color="#000" fill="none" stroke="#030303" stroke-linecap="round" stroke-opacity=".5"/>
    ///   <path d="M27.78 24.9a1.1 1.1 0 0 1-1.08 1.08 1.1 1.1 0 0 1-1.1-1.08 1.1 1.1 0 0 1 1.1-1.1 1.1 1.1 0 0 1 1.08 1.1" color="#000" fill="#fff"/>
    ///   <path d="M45.4 5.14a1.1 1.1 0 0 1-1.08 1.1 1.1 1.1 0 0 1-1.1-1.1 1.1 1.1 0 0 1 1.1-1.08 1.1 1.1 0 0 1 1.1 1.08" color="#000" fill="none" stroke="#030303" stroke-linecap="round" stroke-opacity=".5"/>
    ///   <path d="M45.4 5.14a1.1 1.1 0 0 1-1.08 1.1 1.1 1.1 0 0 1-1.1-1.1 1.1 1.1 0 0 1 1.1-1.08 1.1 1.1 0 0 1 1.1 1.08" color="#000" fill="#fff"/>
    ///   <path d="M28.67 13.8a1.1 1.1 0 0 1-1.1 1.08 1.1 1.1 0 0 1-1.08-1.08 1.1 1.1 0 0 1 1.08-1.1 1.1 1.1 0 0 1 1.1 1.1" color="#000" fill="none" stroke="#030303" stroke-linecap="round" stroke-opacity=".5"/>
    ///   <path d="M28.67 13.8a1.1 1.1 0 0 1-1.1 1.08 1.1 1.1 0 0 1-1.08-1.08 1.1 1.1 0 0 1 1.08-1.1 1.1 1.1 0 0 1 1.1 1.1" color="#000" fill="#fff"/>
    ///   <path d="M35 7.32a1.1 1.1 0 0 1-1.1 1.1 1.1 1.1 0 0 1-1.08-1.1 1.1 1.1 0 0 1 1.1-1.1A1.1 1.1 0 0 1 35 7.33" color="#000" fill="none" stroke="#030303" stroke-linecap="round" stroke-opacity=".5"/>
    ///   <path d="M35 7.32a1.1 1.1 0 0 1-1.1 1.1 1.1 1.1 0 0 1-1.08-1.1 1.1 1.1 0 0 1 1.1-1.1A1.1 1.1 0 0 1 35 7.33" color="#000" fill="#fff"/>
    ///   <text style="line-height:6.61458302px" x="35.74" y="284.49" font-size="5.29" font-family="Sans" letter-spacing="0" word-spacing="0" fill="#b3b3b3" stroke-width=".26" transform="translate(19.595 -267)">
    ///     <tspan x="35.74" y="284.49" font-size="10.58">→</tspan>
    ///   </text>
    /// </svg>
    ///
    /// The tolerance value controls the maximum distance between the curved input
    /// segments and their polyline approximations. (In technical terms, this is the
    /// Hausdorff distance). The algorithm attempts to bound this distance between
    /// by `tolerance` but this is not absolutely guaranteed. The appropriate value
    /// depends on the use, but for antialiased rendering, a value of 0.25 has been
    /// determined to give good results. The number of segments tends to scale as the
    /// inverse square root of tolerance.
    ///
    /// <svg viewBox="0 0 47.5 13.2" height="100" width="350" xmlns="http://www.w3.org/2000/svg">
    ///   <path d="M-2.44 9.53c16.27-8.5 39.68-7.93 52.13 1.9" fill="none" stroke="#dde9af" stroke-width="4.6"/>
    ///   <path d="M-1.97 9.3C14.28 1.03 37.36 1.7 49.7 11.4" fill="none" stroke="#00d400" stroke-width=".57" stroke-linecap="round" stroke-dasharray="4.6, 2.291434"/>
    ///   <path d="M-1.94 10.46L6.2 6.08l28.32-1.4 15.17 6.74" fill="none" stroke="#000" stroke-width=".6"/>
    ///   <path d="M6.83 6.57a.9.9 0 0 1-1.25.15.9.9 0 0 1-.15-1.25.9.9 0 0 1 1.25-.15.9.9 0 0 1 .15 1.25" color="#000" stroke="#000" stroke-width=".57" stroke-linecap="round" stroke-opacity=".5"/>
    ///   <path d="M35.35 5.3a.9.9 0 0 1-1.25.15.9.9 0 0 1-.15-1.25.9.9 0 0 1 1.25-.15.9.9 0 0 1 .15 1.24" color="#000" stroke="#000" stroke-width=".6" stroke-opacity=".5"/>
    ///   <g fill="none" stroke="#ff7f2a" stroke-width=".26">
    ///     <path d="M20.4 3.8l.1 1.83M19.9 4.28l.48-.56.57.52M21.02 5.18l-.5.56-.6-.53" stroke-width=".2978872"/>
    ///   </g>
    /// </svg>
    ///
    /// The callback will be called in order with each element of the generated
    /// path. Because the result is made of polylines, these will be straight-line
    /// path elements only, no curves.
    ///
    /// This algorithm is based on the blog post [Flattening quadratic Béziers]
    /// but with some refinements. For one, there is a more careful approximation
    /// at cusps. For two, the algorithm is extended to work with cubic Béziers
    /// as well, by first subdividing into quadratics and then computing the
    /// subdivision of each quadratic. However, as a clever trick, these quadratics
    /// are subdivided fractionally, and their endpoints are not included.
    ///
    /// TODO: write a paper explaining this in more detail.
    ///
    /// Note: the [`flatten`] function provides the same
    /// functionality but works with slices and other [`PathEl`] iterators.
    ///
    /// [Flattening quadratic Béziers]: https://raphlinus.github.io/graphics/curves/2019/12/23/flatten-quadbez.html
    pub fn flatten(&self, tolerance: f64, callback: impl FnMut(PathEl)) {
        flatten(self, tolerance, callback);
    }

    /// Get the segment at the given element index.
    ///
    /// If you need to access all segments, [`segments`] provides a better
    /// API. This is intended for random access of specific elements, for clients
    /// that require this specifically.
    ///
    /// **note**: This returns the segment that ends at the provided element
    /// index. In effect this means it is *1-indexed*: since no segment ends at
    /// the first element (which is presumed to be a `MoveTo`) `get_seg(0)` will
    /// always return `None`.
    pub fn get_seg(&self, ix: usize) -> Option<PathSeg> {
        if ix == 0 || ix >= self.0.len() {
            return None;
        }
        let last = match self.0[ix - 1].end() {
            Some(pt) => pt,
            None => {
                return None;
            }
        };
        match self.0[ix] {
            PathEl::LineTo(p) => Some(PathSeg::Line(Line::new(last, p))),
            PathEl::QuadTo(p1, p2) => Some(PathSeg::Quad(QuadBez::new(last, p1, p2))),
            PathEl::CurveTo(p1, p2, p3) => Some(PathSeg::Cubic(CubicBez::new(last, p1, p2, p3))),
            PathEl::ClosePath => match self.get_seg_end(ix) {
                Some(pt) => {
                    if last != pt {
                        Some(PathSeg::Line(Line::new(last, pt)))
                    } else {
                        None
                    }
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// Get the end point of the segment if we can calculate it
    ///
    /// The element index counts [`PathEl`] elements, so
    /// for example includes an initial `Moveto`.
    pub fn get_seg_end(&self, ix: usize) -> Option<Point> {
        if ix >= self.0.len() {
            return None;
        }

        match self.0[ix] {
            PathEl::MoveTo(p) => Some(p),
            PathEl::LineTo(p) => Some(p),
            PathEl::QuadTo(_, p) => Some(p),
            PathEl::CurveTo(_, _, p) => Some(p),
            PathEl::ClosePath => self.0[..ix].iter().rev().find_map(|el| match *el {
                PathEl::MoveTo(p) => Some(p),
                _ => {
                    if !self.0.is_empty() {
                        self.0[0].end()
                    } else {
                        None
                    }
                }
            }),
        }
    }

    /// Returns `true` if the path contains no segments.
    pub fn is_empty(&self) -> bool {
        self.0
            .iter()
            .all(|el| matches!(el, PathEl::MoveTo(..) | PathEl::ClosePath))
    }

    /// Apply an affine transform to the path.
    pub fn apply_affine(&mut self, affine: Affine) {
        for el in self.0.iter_mut() {
            *el = affine * (*el);
        }
    }

    /// Splits the path into multiple paths wherever a move-to happens
    pub fn split_at_moves(&self) -> Vec<BezPath> {
        if self.0.is_empty() {
            return Vec::<BezPath>::new(); // empty path
        }

        // Split by move_to's
        self.0
            .iter()
            .skip(1) // Skip first element since it's always treated as a move-to
            .enumerate() // Get index of each move-to
            .filter(|el| match el.1 {
                PathEl::MoveTo(_) => true,
                _ => false,
            })
            .map(|(i, _)| i + 1)
            .chain(vec![self.0.len()].into_iter()) // Append the size of the element list
            .scan(0, |start, end| {
                // Get start/end index pairs
                let res = (*start, end);
                *start = end;
                Some(res)
            })
            .filter(|(start, end)| (end - start) > 1) // Don't include duplicate move-tos
            .map(|(start, end)| BezPath::from_vec(self.0[start..end].to_vec())) // Create new paths based on start/end bounds
            .collect::<Vec<_>>()
    }

    /// Closes all subpaths
    pub fn close_subpaths(&mut self) {
        if self.0.is_empty() {
            return; // No subpaths to close
        }

        /// Gets the start of the subpath from the given element iterator
        fn find_subpath_start(path: &BezPath, index: usize) -> Point {
            match path.0[..index].iter().rev().find_map(|el| match *el {
                PathEl::MoveTo(start) => Some(start),
                _ => None,
            }) {
                Some(point) => point,
                None => {
                    // This can happen when the path doesn't start with a move-to like it should
                    match path.0.get(0) {
                        Some(el) => el
                            .end()
                            .expect("Can't get end point of an element in this function!"),
                        _ => panic!("Path is empty when it shouldn't be!"),
                    }
                }
            }
        }

        // Find move-tos and then close subpaths
        let insert_positions = self
            .0
            .iter()
            .skip(1) // Skip first element since it's always treated as a move-to
            .enumerate() // Get index of each move-to
            .filter(|el| match el.1 {
                PathEl::MoveTo(_) => true,
                _ => false,
            })
            .map(|(i, _)| i + 1)
            .chain(vec![self.0.len()].into_iter()) // Append the size of the element list
            .filter(|i| match self.0[i - 1] {
                // Filter out subpaths that are already closed
                PathEl::ClosePath => false,
                _ => true,
            })
            .map(|i| {
                // Find start/end points of subpaths
                (
                    i,
                    (
                        find_subpath_start(self, i),
                        self.0[i - 1]
                            .end()
                            .expect("Can't get end point of an element in this function!"),
                    ),
                )
            })
            .filter_map(|(i, (start, end))| if start != end { Some(i) } else { None }) // Find which subpaths are implicitly closed already
            .rev() // Insert the close-paths in reverse order
            .collect::<Vec<_>>();
        insert_positions
            .iter()
            .for_each(|i| self.0.insert(*i, PathEl::ClosePath));
    }

    /// Gets the index and t-parameter of each segment in the path that intersects
    /// with another segment in the path
    pub fn self_intersections(&self, accuracy: f64) -> Vec<((usize, f64), (usize, f64))> {
        let segments = self
            .iter()
            .enumerate()
            .filter_map(|(index, _)| match self.get_seg(index) {
                Some(seg) => Some((index, seg)),
                None => None,
            })
            .collect::<Vec<_>>();

        // Find self-intersecting segments
        segments
            .iter()
            .by_ref()
            .map(|&(index, seg)| {
                seg.self_intersections(accuracy)
                    .into_iter() // Get self-intersections for each segment
                    .map(|intersect| ((index, intersect.0), (index, intersect.1)))
                    .chain(
                        segments
                            .iter()
                            .filter(|&(index2, _)| *index2 > index)
                            .map(|&(index2, seg2)| {
                                let flags = if index == 1 && index2 == self.0.len() - 1 {
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
                                //curve_curve_intersections(&seg, &seg2, flags, accuracy).into_iter() // Get intersections with segments after this one
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

    /// Break paths segments at the specified self-intersection points by adding move-tos and snaps those points together
    pub fn break_at_self_intersections(
        path: &mut BezPath,
        intersections: &Vec<((usize, f64), (usize, f64))>,
    ) {
        if path.0.is_empty() {
            return; // No path to break
        }

        if intersections.is_empty() {
            return; // No intersections to break at
        }

        // Convert the list of elements into a vector of tuples to make tracking indices easier
        let mut elements = path
            .elements()
            .iter()
            .enumerate()
            .map(|(index, el)| (el, vec![(path.get_seg(index), (0.0..1.0))]))
            .collect::<Vec<_>>();

        // Split the segments at each intersection
        intersections
            .iter()
            .for_each(|&((index1, t1), (index2, t2))| {
                fn find_segment_index(
                    segs: &Vec<(Option<PathSeg>, Range<f64>)>,
                    t: f64,
                ) -> Option<(usize, (PathSeg, Range<f64>))> {
                    // Find which range contains the t parameter
                    segs.iter()
                        .enumerate()
                        .filter(|(_, (seg, _))| seg.is_some()) // Ignore elements that cannot be converted to segments
                        .find(|(_, (_, range))| range.contains(&t)) // Search for the range containing t
                        .map(|(index, (seg, range))| (index, (seg.unwrap(), range.clone())))
                }

                // Split segments at each intersection
                fn modify_segment_list(
                    segs: &mut Vec<(Option<PathSeg>, Range<f64>)>,
                    t: f64,
                ) -> usize {
                    let (seg_index, (seg, range)) =
                        find_segment_index(&segs, t).expect("Intersection segment not found!");
                    let t_mapped = (t - range.start) / (range.end - range.start);

                    // Replace segment with 2 subsegments
                    segs.splice(
                        seg_index..(seg_index + 1),
                        vec![
                            (Some(seg.subsegment(0.0..t_mapped)), (range.start..t)),
                            (Some(seg.subsegment(t_mapped..1.0)), (t..range.end)),
                        ],
                    );

                    seg_index
                }
                let seg_index1 = modify_segment_list(
                    &mut elements
                        .get_mut(index1)
                        .expect("Intersection index out of range!")
                        .1,
                    t1,
                );
                let seg_index2 = modify_segment_list(
                    &mut elements
                        .get_mut(index2)
                        .expect("Intersection index out of range!")
                        .1,
                    t2,
                );

                // If we are splitting the same segment, and t2 < t1, seg_index1 needs to be adjusted
                let seg_index1 = if index1 == index2 && t2 < t1 {
                    seg_index1 + 1
                } else {
                    seg_index1
                };

                // Snap intersection points to be exactly the same
                let pt_snap = ((elements[index1].1[seg_index1].0.unwrap().end().to_vec2()
                    + elements[index2].1[seg_index2].0.unwrap().end().to_vec2())
                    / 2.)
                    .to_point();
                elements[index1].1[seg_index1].0.unwrap().set_end(pt_snap);
                elements[index1].1[seg_index1 + 1]
                    .0
                    .unwrap()
                    .set_start(pt_snap);
                elements[index2].1[seg_index2].0.unwrap().set_end(pt_snap);
                elements[index2].1[seg_index2 + 1]
                    .0
                    .unwrap()
                    .set_start(pt_snap);
            });

        // Create a new list of elements
        path.0 = elements
            .into_iter()
            .map(|(el, segs)| {
                // Convert list of segments into list of elements
                match el {
                    PathEl::MoveTo(_) => vec![*el], // No need to convert the element
                    _ => {
                        segs.iter()
                            .filter_map(|(seg, _)| *seg) // Discard the range and unwrap the segments
                            .map(|seg| {
                                // Convert to path elements
                                let el = match seg {
                                    PathSeg::Line(line) => PathEl::LineTo(line.p1),
                                    PathSeg::Quad(quad) => PathEl::QuadTo(quad.p1, quad.p2),
                                    PathSeg::Cubic(cubic) => {
                                        PathEl::CurveTo(cubic.p1, cubic.p2, cubic.p3)
                                    }
                                };
                                vec![PathEl::MoveTo(seg.start()), el]
                            })
                            .flatten()
                            .skip(1) // Skip the extra move-to added at the beginning
                            .collect::<Vec<_>>()
                    }
                }
            })
            .flatten() // Flatten into one list of segments
            .collect::<Vec<_>>();
    }

    /// Break paths segments at the specified intersection points by adding move-tos and snaps those points together
    pub fn break_at_intersections(
        path1: &mut BezPath,
        path2: &mut BezPath,
        intersections: &Vec<((usize, f64), (usize, f64))>,
    ) {
        if path1.0.is_empty() || path2.0.is_empty() {
            return; // No paths to break
        }

        if intersections.is_empty() {
            return; // No intersections to break at
        }

        // Convert the list of elements into a vector of tuples to make tracking indices easier
        let mut elements1 = path1
            .elements()
            .iter()
            .enumerate()
            .map(|(index, el)| (el, vec![(path1.get_seg(index), (0.0..1.0))]))
            .collect::<Vec<_>>();
        let mut elements2 = path2
            .elements()
            .iter()
            .enumerate()
            .map(|(index, el)| (el, vec![(path2.get_seg(index), (0.0..1.0))]))
            .collect::<Vec<_>>();

        // Split the segments at each intersection
        intersections
            .iter()
            .for_each(|&((index1, t1), (index2, t2))| {
                fn find_segment_index(
                    segs: &Vec<(Option<PathSeg>, Range<f64>)>,
                    t: f64,
                ) -> Option<(usize, (PathSeg, Range<f64>))> {
                    // Find which range contains the t parameter
                    segs.iter()
                        .enumerate()
                        .filter(|(_, (seg, _))| seg.is_some()) // Ignore elements that cannot be converted to segments
                        .find(|(_, (_, range))| range.contains(&t)) // Search for the range containing t
                        .map(|(index, (seg, range))| (index, (seg.unwrap(), range.clone())))
                }

                // Split segments at each intersection
                fn modify_segment_list(
                    segs: &mut Vec<(Option<PathSeg>, Range<f64>)>,
                    t: f64,
                ) -> usize {
                    let (seg_index, (seg, range)) =
                        find_segment_index(&segs, t).expect("Intersection segment not found!");
                    let t_mapped = (t - range.start) / (range.end - range.start);

                    // Replace segment with 2 subsegments
                    segs.splice(
                        seg_index..(seg_index + 1),
                        vec![
                            (Some(seg.subsegment(0.0..t_mapped)), (range.start..t)),
                            (Some(seg.subsegment(t_mapped..1.0)), (t..range.end)),
                        ],
                    );

                    seg_index
                }
                let seg_index1 = modify_segment_list(
                    &mut elements1
                        .get_mut(index1)
                        .expect("Intersection index out of range!")
                        .1,
                    t1,
                );
                let seg_index2 = modify_segment_list(
                    &mut elements2
                        .get_mut(index2)
                        .expect("Intersection index out of range!")
                        .1,
                    t2,
                );

                // If we are splitting the same segment, and t2 < t1, seg_index1 needs to be adjusted
                let seg_index1 = if index1 == index2 && t2 < t1 {
                    seg_index1 + 1
                } else {
                    seg_index1
                };

                // Snap intersection points to be exactly the same
                let pt_snap = ((elements1[index1].1[seg_index1].0.unwrap().end().to_vec2()
                    + elements2[index2].1[seg_index2].0.unwrap().end().to_vec2())
                    / 2.)
                    .to_point();
                elements1[index1].1[seg_index1].0.unwrap().set_end(pt_snap);
                elements1[index1].1[seg_index1 + 1]
                    .0
                    .unwrap()
                    .set_start(pt_snap);
                elements2[index2].1[seg_index2].0.unwrap().set_end(pt_snap);
                elements2[index2].1[seg_index2 + 1]
                    .0
                    .unwrap()
                    .set_start(pt_snap);
            });

        // Create a new list of elements
        path1.0 = elements1
            .into_iter()
            .map(|(el, segs)| {
                // Convert list of segments into list of elements
                match el {
                    PathEl::MoveTo(_) => vec![*el], // No need to convert the element
                    _ => {
                        segs.iter()
                            .filter_map(|(seg, _)| *seg) // Discard the range and unwrap the segments
                            .map(|seg| {
                                // Convert to path elements
                                let el = match seg {
                                    PathSeg::Line(line) => PathEl::LineTo(line.p1),
                                    PathSeg::Quad(quad) => PathEl::QuadTo(quad.p1, quad.p2),
                                    PathSeg::Cubic(cubic) => {
                                        PathEl::CurveTo(cubic.p1, cubic.p2, cubic.p3)
                                    }
                                };
                                vec![PathEl::MoveTo(seg.start()), el]
                            })
                            .flatten()
                            .skip(1) // Skip the extra move-to added at the beginning
                            .collect::<Vec<_>>()
                    }
                }
            })
            .flatten() // Flatten into one list of segments
            .collect::<Vec<_>>();
        path2.0 = elements2
            .into_iter()
            .map(|(el, segs)| {
                // Convert list of segments into list of elements
                match el {
                    PathEl::MoveTo(_) => vec![*el], // No need to convert the element
                    _ => {
                        segs.iter()
                            .filter_map(|(seg, _)| *seg) // Discard the range and unwrap the segments
                            .map(|seg| {
                                // Convert to path elements
                                let el = match seg {
                                    PathSeg::Line(line) => PathEl::LineTo(line.p1),
                                    PathSeg::Quad(quad) => PathEl::QuadTo(quad.p1, quad.p2),
                                    PathSeg::Cubic(cubic) => {
                                        PathEl::CurveTo(cubic.p1, cubic.p2, cubic.p3)
                                    }
                                };
                                vec![PathEl::MoveTo(seg.start()), el]
                            })
                            .flatten()
                            .skip(1) // Skip the extra move-to added at the beginning
                            .collect::<Vec<_>>()
                    }
                }
            })
            .flatten() // Flatten into one list of segments
            .collect::<Vec<_>>();
    }

    /// Is this path finite?
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.0.iter().all(|v| v.is_finite())
    }

    /// Is this path NaN?
    #[inline]
    pub fn is_nan(&self) -> bool {
        self.0.iter().any(|v| v.is_nan())
    }
}

impl FromIterator<PathEl> for BezPath {
    fn from_iter<T: IntoIterator<Item = PathEl>>(iter: T) -> Self {
        let el_vec: Vec<_> = iter.into_iter().collect();
        BezPath::from_vec(el_vec)
    }
}

/// Allow iteration over references to `BezPath`.
///
/// Note: the semantics are slightly different from simply iterating over the
/// slice, as it returns `PathEl` items, rather than references.
impl<'a> IntoIterator for &'a BezPath {
    type Item = PathEl;
    type IntoIter = std::iter::Cloned<std::slice::Iter<'a, PathEl>>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements().iter().cloned()
    }
}

impl IntoIterator for BezPath {
    type Item = PathEl;
    type IntoIter = std::vec::IntoIter<PathEl>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl Extend<PathEl> for BezPath {
    fn extend<I: IntoIterator<Item = PathEl>>(&mut self, iter: I) {
        self.0.extend(iter);
    }
}

/// Proportion of tolerance budget that goes to cubic to quadratic conversion.
const TO_QUAD_TOL: f64 = 0.1;

/// Flatten the path, invoking the callback repeatedly.
///
/// See [`BezPath::flatten`] for more discussion.
/// This signature is a bit more general, allowing flattening of `&[PathEl]` slices
/// and other iterators yielding `PathEl`.
pub fn flatten(
    path: impl IntoIterator<Item = PathEl>,
    tolerance: f64,
    mut callback: impl FnMut(PathEl),
) {
    let sqrt_tol = tolerance.sqrt();
    let mut last_pt = None;
    let mut quad_buf = Vec::new();
    for el in path {
        match el {
            PathEl::MoveTo(p) => {
                last_pt = Some(p);
                callback(PathEl::MoveTo(p));
            }
            PathEl::LineTo(p) => {
                last_pt = Some(p);
                callback(PathEl::LineTo(p));
            }
            PathEl::QuadTo(p1, p2) => {
                if let Some(p0) = last_pt {
                    let q = QuadBez::new(p0, p1, p2);
                    let params = q.estimate_subdiv(sqrt_tol);
                    let n = ((0.5 * params.val / sqrt_tol).ceil() as usize).max(1);
                    let step = 1.0 / (n as f64);
                    for i in 1..n {
                        let u = (i as f64) * step;
                        let t = q.determine_subdiv_t(&params, u);
                        let p = q.eval(t);
                        callback(PathEl::LineTo(p));
                    }
                    callback(PathEl::LineTo(p2));
                }
                last_pt = Some(p2);
            }
            PathEl::CurveTo(p1, p2, p3) => {
                if let Some(p0) = last_pt {
                    let c = CubicBez::new(p0, p1, p2, p3);

                    // Subdivide into quadratics, and estimate the number of
                    // subdivisions required for each, summing to arrive at an
                    // estimate for the number of subdivisions for the cubic.
                    // Also retain these parameters for later.
                    let iter = c.to_quads(tolerance * TO_QUAD_TOL);
                    quad_buf.clear();
                    quad_buf.reserve(iter.size_hint().0);
                    let sqrt_remain_tol = sqrt_tol * (1.0 - TO_QUAD_TOL).sqrt();
                    let mut sum = 0.0;
                    for (_, _, q) in iter {
                        let params = q.estimate_subdiv(sqrt_remain_tol);
                        sum += params.val;
                        quad_buf.push((q, params));
                    }
                    let n = ((0.5 * sum / sqrt_remain_tol).ceil() as usize).max(1);

                    // Iterate through the quadratics, outputting the points of
                    // subdivisions that fall within that quadratic.
                    let step = sum / (n as f64);
                    let mut i = 1;
                    let mut val_sum = 0.0;
                    for (q, params) in &quad_buf {
                        let mut target = (i as f64) * step;
                        let recip_val = params.val.recip();
                        while target < val_sum + params.val {
                            let u = (target - val_sum) * recip_val;
                            let t = q.determine_subdiv_t(params, u);
                            let p = q.eval(t);
                            callback(PathEl::LineTo(p));
                            i += 1;
                            if i == n + 1 {
                                break;
                            }
                            target = (i as f64) * step;
                        }
                        val_sum += params.val;
                    }
                    callback(PathEl::LineTo(p3));
                }
                last_pt = Some(p3);
            }
            PathEl::ClosePath => {
                last_pt = None;
                callback(PathEl::ClosePath);
            }
        }
    }
}

impl Mul<PathEl> for Affine {
    type Output = PathEl;

    fn mul(self, other: PathEl) -> PathEl {
        match other {
            PathEl::MoveTo(p) => PathEl::MoveTo(self * p),
            PathEl::LineTo(p) => PathEl::LineTo(self * p),
            PathEl::QuadTo(p1, p2) => PathEl::QuadTo(self * p1, self * p2),
            PathEl::CurveTo(p1, p2, p3) => PathEl::CurveTo(self * p1, self * p2, self * p3),
            PathEl::ClosePath => PathEl::ClosePath,
        }
    }
}

impl Mul<PathSeg> for Affine {
    type Output = PathSeg;

    fn mul(self, other: PathSeg) -> PathSeg {
        match other {
            PathSeg::Line(line) => PathSeg::Line(self * line),
            PathSeg::Quad(quad) => PathSeg::Quad(self * quad),
            PathSeg::Cubic(cubic) => PathSeg::Cubic(self * cubic),
        }
    }
}

impl Mul<PathSegDeriv> for Affine {
    type Output = PathSegDeriv;

    fn mul(self, other: PathSegDeriv) -> PathSegDeriv {
        match other {
            PathSegDeriv::Point(point) => PathSegDeriv::Point(self * point),
            PathSegDeriv::Line(line) => PathSegDeriv::Line(self * line),
            PathSegDeriv::Quad(quad) => PathSegDeriv::Quad(self * quad),
        }
    }
}

impl Mul<BezPath> for Affine {
    type Output = BezPath;

    fn mul(self, other: BezPath) -> BezPath {
        BezPath(other.0.iter().map(|&el| self * el).collect())
    }
}

impl<'a> Mul<&'a BezPath> for Affine {
    type Output = BezPath;

    fn mul(self, other: &BezPath) -> BezPath {
        BezPath(other.0.iter().map(|&el| self * el).collect())
    }
}

impl Mul<PathEl> for TranslateScale {
    type Output = PathEl;

    fn mul(self, other: PathEl) -> PathEl {
        match other {
            PathEl::MoveTo(p) => PathEl::MoveTo(self * p),
            PathEl::LineTo(p) => PathEl::LineTo(self * p),
            PathEl::QuadTo(p1, p2) => PathEl::QuadTo(self * p1, self * p2),
            PathEl::CurveTo(p1, p2, p3) => PathEl::CurveTo(self * p1, self * p2, self * p3),
            PathEl::ClosePath => PathEl::ClosePath,
        }
    }
}

impl Mul<PathSeg> for TranslateScale {
    type Output = PathSeg;

    fn mul(self, other: PathSeg) -> PathSeg {
        match other {
            PathSeg::Line(line) => PathSeg::Line(self * line),
            PathSeg::Quad(quad) => PathSeg::Quad(self * quad),
            PathSeg::Cubic(cubic) => PathSeg::Cubic(self * cubic),
        }
    }
}

impl Mul<BezPath> for TranslateScale {
    type Output = BezPath;

    fn mul(self, other: BezPath) -> BezPath {
        BezPath(other.0.iter().map(|&el| self * el).collect())
    }
}

impl<'a> Mul<&'a BezPath> for TranslateScale {
    type Output = BezPath;

    fn mul(self, other: &BezPath) -> BezPath {
        BezPath(other.0.iter().map(|&el| self * el).collect())
    }
}

/// Transform an iterator over path elements into one over path
/// segments.
///
/// See also [`BezPath::segments`].
/// This signature is a bit more general, allowing `&[PathEl]` slices
/// and other iterators yielding `PathEl`.
pub fn segments<I>(elements: I) -> Segments<I::IntoIter>
where
    I: IntoIterator<Item = PathEl>,
{
    Segments {
        elements: elements.into_iter(),
        start_last: None,
    }
}

/// An iterator that transforms path elements to path segments.
///
/// This struct is created by the [`segments`] function.
pub struct Segments<I: Iterator<Item = PathEl>> {
    elements: I,
    start_last: Option<(Point, Point)>,
}

impl<I: Iterator<Item = PathEl>> Iterator for Segments<I> {
    type Item = PathSeg;

    fn next(&mut self) -> Option<PathSeg> {
        for el in &mut self.elements {
            // We first need to check whether this is the first
            // path element we see to fill in the start position.
            let (start, last) = self.start_last.get_or_insert_with(|| {
                let point = el.end().expect("Can't start a segment on a ClosePath");
                (point, point)
            });

            return Some(match el {
                PathEl::MoveTo(p) => {
                    *start = p;
                    *last = p;
                    continue;
                }
                PathEl::LineTo(p) => PathSeg::Line(Line::new(mem::replace(last, p), p)),
                PathEl::QuadTo(p1, p2) => {
                    PathSeg::Quad(QuadBez::new(mem::replace(last, p2), p1, p2))
                }
                PathEl::CurveTo(p1, p2, p3) => {
                    PathSeg::Cubic(CubicBez::new(mem::replace(last, p3), p1, p2, p3))
                }
                PathEl::ClosePath => {
                    if *last != *start {
                        PathSeg::Line(Line::new(mem::replace(last, *start), *start))
                    } else {
                        continue;
                    }
                }
            });
        }

        None
    }
}

impl<I: Iterator<Item = PathEl>> Segments<I> {
    /// Here, `accuracy` specifies the accuracy for each Bézier segment. At worst,
    /// the total error is `accuracy` times the number of Bézier segments.

    // TODO: pub? Or is this subsumed by method of &[PathEl]?
    pub(crate) fn perimeter(self, accuracy: f64) -> f64 {
        self.map(|seg| seg.arclen(accuracy)).sum()
    }

    // Same
    pub(crate) fn area(self) -> f64 {
        self.map(|seg| seg.signed_area()).sum()
    }

    // Same
    pub(crate) fn winding(self, p: Point) -> i32 {
        self.map(|seg| seg.winding(p)).sum()
    }

    // Same
    pub(crate) fn bounding_box(self) -> Rect {
        let mut bbox: Option<Rect> = None;
        for seg in self {
            let seg_bb = ParamCurveExtrema::bounding_box(&seg);
            if let Some(bb) = bbox {
                bbox = Some(bb.union(seg_bb));
            } else {
                bbox = Some(seg_bb)
            }
        }
        bbox.unwrap_or_default()
    }
}

impl ParamCurve for PathSeg {
    fn eval(&self, t: f64) -> Point {
        match *self {
            PathSeg::Line(line) => line.eval(t),
            PathSeg::Quad(quad) => quad.eval(t),
            PathSeg::Cubic(cubic) => cubic.eval(t),
        }
    }

    fn subsegment(&self, range: Range<f64>) -> PathSeg {
        match *self {
            PathSeg::Line(line) => PathSeg::Line(line.subsegment(range)),
            PathSeg::Quad(quad) => PathSeg::Quad(quad.subsegment(range)),
            PathSeg::Cubic(cubic) => PathSeg::Cubic(cubic.subsegment(range)),
        }
    }

    fn set_start(&mut self, pt: Point) {
        match *self {
            PathSeg::Line(mut line) => line.set_start(pt),
            PathSeg::Quad(mut quad) => quad.set_start(pt),
            PathSeg::Cubic(mut cubic) => cubic.set_start(pt),
        }
    }

    fn set_end(&mut self, pt: Point) {
        match *self {
            PathSeg::Line(mut line) => line.set_end(pt),
            PathSeg::Quad(mut quad) => quad.set_end(pt),
            PathSeg::Cubic(mut cubic) => cubic.set_end(pt),
        }
    }

    #[inline]
    fn get_affine_transformed(&self, affine: &Affine) -> Self {
        *affine * *self
    }
}

impl ParamCurve for PathSegDeriv {
    fn eval(&self, t: f64) -> Point {
        match *self {
            PathSegDeriv::Point(point) => point.eval(t),
            PathSegDeriv::Line(line) => line.eval(t),
            PathSegDeriv::Quad(quad) => quad.eval(t),
        }
    }

    fn subsegment(&self, range: Range<f64>) -> PathSegDeriv {
        match *self {
            PathSegDeriv::Point(point) => PathSegDeriv::Point(point.subsegment(range)),
            PathSegDeriv::Line(line) => PathSegDeriv::Line(line.subsegment(range)),
            PathSegDeriv::Quad(quad) => PathSegDeriv::Quad(quad.subsegment(range)),
        }
    }

    fn set_start(&mut self, pt: Point) {
        match *self {
            PathSegDeriv::Point(mut point) => point.set_start(pt),
            PathSegDeriv::Line(mut line) => line.set_start(pt),
            PathSegDeriv::Quad(mut quad) => quad.set_start(pt),
        }
    }

    fn set_end(&mut self, pt: Point) {
        match *self {
            PathSegDeriv::Point(mut point) => point.set_end(pt),
            PathSegDeriv::Line(mut line) => line.set_end(pt),
            PathSegDeriv::Quad(mut quad) => quad.set_end(pt),
        }
    }

    // #[inline]
    fn get_affine_transformed(&self, affine: &Affine) -> Self {
        *affine * *self
    }
}

impl ParamCurveDeriv for PathSeg {
    type DerivResult = PathSegDeriv;

    fn deriv(&self) -> PathSegDeriv {
        match *self {
            PathSeg::Line(line) => PathSegDeriv::Point(line.deriv()),
            PathSeg::Quad(quad) => PathSegDeriv::Line(quad.deriv()),
            PathSeg::Cubic(cubic) => PathSegDeriv::Quad(cubic.deriv()),
        }
    }
}

impl ParamCurveArclen for PathSeg {
    fn arclen(&self, accuracy: f64) -> f64 {
        match *self {
            PathSeg::Line(line) => line.arclen(accuracy),
            PathSeg::Quad(quad) => quad.arclen(accuracy),
            PathSeg::Cubic(cubic) => cubic.arclen(accuracy),
        }
    }

    fn inv_arclen(&self, arclen: f64, accuracy: f64) -> f64 {
        match *self {
            PathSeg::Line(line) => line.inv_arclen(arclen, accuracy),
            PathSeg::Quad(quad) => quad.inv_arclen(arclen, accuracy),
            PathSeg::Cubic(cubic) => cubic.inv_arclen(arclen, accuracy),
        }
    }
}

impl ParamCurveArea for PathSeg {
    fn signed_area(&self) -> f64 {
        match *self {
            PathSeg::Line(line) => line.signed_area(),
            PathSeg::Quad(quad) => quad.signed_area(),
            PathSeg::Cubic(cubic) => cubic.signed_area(),
        }
    }
}

impl ParamCurveNearest for PathSeg {
    fn nearest(&self, p: Point, accuracy: f64) -> Nearest {
        match *self {
            PathSeg::Line(line) => line.nearest(p, accuracy),
            PathSeg::Quad(quad) => quad.nearest(p, accuracy),
            PathSeg::Cubic(cubic) => cubic.nearest(p, accuracy),
        }
    }
}

impl ParamCurveExtrema for PathSeg {
    fn extrema_x(&self) -> ArrayVec<f64, MAX_EXTREMA> {
        match *self {
            PathSeg::Line(line) => line.extrema_x(),
            PathSeg::Quad(quad) => quad.extrema_x(),
            PathSeg::Cubic(cubic) => cubic.extrema_x(),
        }
    }

    fn extrema_y(&self) -> ArrayVec<f64, MAX_EXTREMA> {
        match *self {
            PathSeg::Line(line) => line.extrema_y(),
            PathSeg::Quad(quad) => quad.extrema_y(),
            PathSeg::Cubic(cubic) => cubic.extrema_y(),
        }
    }
}

impl PathSeg {
    /// Returns a new `PathSeg` describing the same path as `self`, but with
    /// the points reversed.
    pub fn reverse(&self) -> PathSeg {
        match self {
            PathSeg::Line(Line { p0, p1 }) => PathSeg::Line(Line::new(*p1, *p0)),
            PathSeg::Quad(q) => PathSeg::Quad(QuadBez::new(q.p2, q.p1, q.p0)),
            PathSeg::Cubic(c) => PathSeg::Cubic(CubicBez::new(c.p3, c.p2, c.p1, c.p0)),
        }
    }

    /// Convert this segment to a cubic bezier.
    pub fn to_cubic(&self) -> CubicBez {
        match *self {
            PathSeg::Line(Line { p0, p1 }) => CubicBez::new(p0, p0, p1, p1),
            PathSeg::Cubic(c) => c,
            PathSeg::Quad(q) => q.raise(),
        }
    }

    // Assumes split at extrema.
    fn winding_inner<T: Fn(Point) -> f64, U: Fn(Point) -> f64>(
        &self,
        p: Point,
        convert: T,
        convert_perp: U,
    ) -> Option<(f64, i32)> {
        let start = self.start();
        let end = self.end();

        let start_par = convert(start);
        let start_perp = convert_perp(start);
        let end_par = convert(end);
        let end_perp = convert_perp(end);
        let p_par = convert(p);
        let p_perp = convert_perp(p);

        let sign = if end_perp > start_perp {
            if p_perp < start_perp || p_perp >= end_perp {
                return None;
            }
            Some(-1)
        } else if end_perp < start_perp {
            if p_perp < end_perp || p_perp >= start_perp {
                return None;
            }
            Some(1)
        } else {
            // This can only happen for coincident lines. We can effectively
            // ignore this situation.
            None
        };

        if sign.is_none() {
            // Self does not intersect with the ray
            return None;
        }
        let sign = sign.unwrap();

        match *self {
            PathSeg::Line(_line) => {
                if p_par < start_par.min(end_par) {
                    // Self does not intersect with the ray
                    return None;
                }

                // Line equation ax + by = c
                let a = end_perp - start_perp;
                let b = start_par - end_par;
                let c = a * start_par + b * start_perp;
                if (a * p_par + b * p_perp - c) * (sign as f64) <= 0.0 {
                    let (c0, c1) = (c / a, -b / a);
                    if c0.is_infinite() || c1.is_infinite() {
                        return None;
                    }
                    Some((c0 + c1 * p_perp, sign))
                } else {
                    None
                }
            }
            PathSeg::Quad(quad) => {
                let p1 = quad.p1;
                let p1_par = convert(p1);

                if p_par < start_par.min(end_par).min(p1_par) {
                    return None;
                }

                // Quadratic equation
                let (a, b, c) = quad.parameters();
                let roots = solve_quadratic(
                    convert_perp(c.to_point()) - p_perp,
                    convert_perp(b.to_point()),
                    convert_perp(a.to_point()),
                )
                .into_iter()
                .filter(|&t| (0.0..=1.0).contains(&t))
                .collect::<Vec<_>>();
                roots.get(0).map(|r| (convert(quad.eval(*r)), sign))
            }
            PathSeg::Cubic(cubic) => {
                let p1 = cubic.p1;
                let p2 = cubic.p2;
                let p1_par = convert(p1);
                let p2_par = convert(p2);

                if p_par < start_par.min(end_par).min(p1_par).min(p2_par) {
                    return None;
                }

                // Cubic equation
                let (a, b, c, d) = cubic.parameters();
                let roots = solve_cubic(
                    convert_perp(d.to_point()) - p_perp,
                    convert_perp(c.to_point()),
                    convert_perp(b.to_point()),
                    convert_perp(a.to_point()),
                )
                .into_iter()
                .filter(|&t| (0.0..=1.0).contains(&t))
                .collect::<Vec<_>>();
                roots.get(0).map(|r| (convert(cubic.eval(*r)), sign))
            }
        }
    }

    /// Compute the winding number contribution of a single segment.
    ///
    /// Cast a ray to the left and count intersections.
    fn winding(&self, p: Point) -> i32 {
        self.winding_x_results(p)
            .into_iter()
            .map(|(_, sign)| sign)
            .sum()
    }

    /// Compute the winding number contribution and points of intersection of a single segment along a horizontal ray.
    ///
    /// Cast a ray to the left and record intersections.
    pub fn winding_x_results(&self, p: Point) -> Vec<(f64, i32)> {
        self.extrema_ranges()
            .into_iter()
            .filter_map(|range| {
                self.subsegment(range)
                    .winding_inner(p, |pt| pt.x, |pt| pt.y)
            })
            .collect()
    }

    /// Compute the winding number contribution and points of intersection of a single segment along a vertical ray.
    ///
    /// Cast a ray to the upwards and record intersections.
    pub fn winding_y_results(&self, p: Point) -> Vec<(f64, i32)> {
        self.extrema_ranges()
            .into_iter()
            .filter_map(|range| {
                self.subsegment(range)
                    .winding_inner(p, |pt| pt.y, |pt| pt.x)
            })
            .collect()
    }

    /// Compute intersections against a line.
    ///
    /// Returns a vector of the intersections. For each intersection,
    /// the `t` value of the segment and line are given.
    ///
    /// Note: This test is designed to be inclusive of points near the endpoints
    /// of the segment. This is so that testing a line against multiple
    /// contiguous segments of a path will be guaranteed to catch at least one
    /// of them. In such cases, use higher level logic to coalesce the hits
    /// (the `t` value may be slightly outside the range of 0..1).
    ///
    /// # Examples
    ///
    /// ```
    /// # use kurbo::*;
    /// let seg = PathSeg::Line(Line::new((0.0, 0.0), (2.0, 0.0)));
    /// let line = Line::new((1.0, 2.0), (1.0, -2.0));
    /// let intersection = seg.intersect_line(line);
    /// assert_eq!(intersection.len(), 1);
    /// let intersection = intersection[0];
    /// assert_eq!(intersection.segment_t, 0.5);
    /// assert_eq!(intersection.line_t, 0.5);
    ///
    /// let point = seg.eval(intersection.segment_t);
    /// assert_eq!(point, Point::new(1.0, 0.0));
    /// ```
    pub fn intersect_line(&self, line: Line) -> ArrayVec<LineIntersection, 3> {
        const EPSILON: f64 = 1e-9;
        let p0 = line.p0;
        let p1 = line.p1;
        let dx = p1.x - p0.x;
        let dy = p1.y - p0.y;
        let mut result = ArrayVec::new();
        match self {
            PathSeg::Line(l) => {
                let det = dx * (l.p1.y - l.p0.y) - dy * (l.p1.x - l.p0.x);
                if det.abs() < EPSILON {
                    // Lines are coincident (or nearly so).
                    return result;
                }
                let t = dx * (p0.y - l.p0.y) - dy * (p0.x - l.p0.x);
                // t = position on self
                let t = t / det;
                if (-EPSILON..=(1.0 + EPSILON)).contains(&t) {
                    // u = position on probe line
                    let u =
                        (l.p0.x - p0.x) * (l.p1.y - l.p0.y) - (l.p0.y - p0.y) * (l.p1.x - l.p0.x);
                    let u = u / det;
                    if (0.0..=1.0).contains(&u) {
                        result.push(LineIntersection::new(u, t));
                    }
                }
            }
            PathSeg::Quad(q) => {
                // The basic technique here is to determine x and y as a quadratic polynomial
                // as a function of t. Then plug those values into the line equation for the
                // probe line (giving a sort of signed distance from the probe line) and solve
                // that for t.
                let (px0, px1, px2) = quadratic_bez_coefs(q.p0.x, q.p1.x, q.p2.x);
                let (py0, py1, py2) = quadratic_bez_coefs(q.p0.y, q.p1.y, q.p2.y);
                let c0 = dy * (px0 - p0.x) - dx * (py0 - p0.y);
                let c1 = dy * px1 - dx * py1;
                let c2 = dy * px2 - dx * py2;
                let invlen2 = (dx * dx + dy * dy).recip();
                for t in crate::common::solve_quadratic(c0, c1, c2) {
                    if (-EPSILON..=(1.0 + EPSILON)).contains(&t) {
                        let x = px0 + t * px1 + t * t * px2;
                        let y = py0 + t * py1 + t * t * py2;
                        let u = ((x - p0.x) * dx + (y - p0.y) * dy) * invlen2;
                        if (0.0..=1.0).contains(&u) {
                            result.push(LineIntersection::new(u, t));
                        }
                    }
                }
            }
            PathSeg::Cubic(c) => {
                // Same technique as above, but cubic polynomial.
                let (px0, px1, px2, px3) = cubic_bez_coefs(c.p0.x, c.p1.x, c.p2.x, c.p3.x);
                let (py0, py1, py2, py3) = cubic_bez_coefs(c.p0.y, c.p1.y, c.p2.y, c.p3.y);
                let c0 = dy * (px0 - p0.x) - dx * (py0 - p0.y);
                let c1 = dy * px1 - dx * py1;
                let c2 = dy * px2 - dx * py2;
                let c3 = dy * px3 - dx * py3;
                let invlen2 = (dx * dx + dy * dy).recip();
                for t in crate::common::solve_cubic(c0, c1, c2, c3) {
                    if (-EPSILON..=(1.0 + EPSILON)).contains(&t) {
                        let x = px0 + t * px1 + t * t * px2 + t * t * t * px3;
                        let y = py0 + t * py1 + t * t * py2 + t * t * t * py3;
                        let u = ((x - p0.x) * dx + (y - p0.y) * dy) * invlen2;
                        if (0.0..=1.0).contains(&u) {
                            result.push(LineIntersection::new(u, t));
                        }
                    }
                }
            }
        }
        result
    }

    /// Gets possible self intersections for a given path segment, ignoring the end
    /// points
    fn self_intersections(&self, accuracy: f64) -> ArrayVec<(f64, f64), 9> {
        match self {
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

    /// Is this Bezier path finite?
    #[inline]
    pub fn is_finite(&self) -> bool {
        match self {
            PathSeg::Line(line) => line.is_finite(),
            PathSeg::Quad(quad_bez) => quad_bez.is_finite(),
            PathSeg::Cubic(cubic_bez) => cubic_bez.is_finite(),
        }
    }

    /// Is this Bezier path NaN?
    #[inline]
    pub fn is_nan(&self) -> bool {
        match self {
            PathSeg::Line(line) => line.is_nan(),
            PathSeg::Quad(quad_bez) => quad_bez.is_nan(),
            PathSeg::Cubic(cubic_bez) => cubic_bez.is_nan(),
        }
    }

    #[inline]
    fn as_vec2_vec(&self) -> ArrayVec<Vec2, 4> {
        let mut a = ArrayVec::new();
        match self {
            PathSeg::Line(l) => {
                a.push(l.p0.to_vec2());
                a.push(l.p1.to_vec2());
            }
            PathSeg::Quad(q) => {
                a.push(q.p0.to_vec2());
                a.push(q.p1.to_vec2());
                a.push(q.p2.to_vec2());
            }
            PathSeg::Cubic(c) => {
                a.push(c.p0.to_vec2());
                a.push(c.p1.to_vec2());
                a.push(c.p2.to_vec2());
                a.push(c.p3.to_vec2());
            }
        };
        a
    }

    /// Minimum distance between two PathSegs
    ///
    /// Returns a tuple of the distance, the path time `t1` of the closest point
    /// on the first PathSeg, and the path time `t2` of the closest point on the
    /// second PathSeg.
    pub fn min_dist(&self, other: PathSeg, accuracy: f64) -> MinDistance {
        let (distance, t1, t2) = crate::mindist::min_dist_param(
            &self.as_vec2_vec(),
            &other.as_vec2_vec(),
            (0.0, 1.0),
            (0.0, 1.0),
            accuracy,
            None,
        );
        MinDistance {
            distance: distance.sqrt(),
            t1,
            t2,
        }
    }
}

impl LineIntersection {
    fn new(line_t: f64, segment_t: f64) -> Self {
        LineIntersection { line_t, segment_t }
    }

    /// Is this line intersection finite?
    #[inline]
    pub fn is_finite(self) -> bool {
        self.line_t.is_finite() && self.segment_t.is_finite()
    }

    /// Is this line intersection NaN?
    #[inline]
    pub fn is_nan(self) -> bool {
        self.line_t.is_nan() || self.segment_t.is_nan()
    }
}

// Return polynomial coefficients given cubic bezier coordinates.
fn quadratic_bez_coefs(x0: f64, x1: f64, x2: f64) -> (f64, f64, f64) {
    let p0 = x0;
    let p1 = 2.0 * x1 - 2.0 * x0;
    let p2 = x2 - 2.0 * x1 + x0;
    (p0, p1, p2)
}

// Return polynomial coefficients given cubic bezier coordinates.
fn cubic_bez_coefs(x0: f64, x1: f64, x2: f64, x3: f64) -> (f64, f64, f64, f64) {
    let p0 = x0;
    let p1 = 3.0 * x1 - 3.0 * x0;
    let p2 = 3.0 * x2 - 6.0 * x1 + 3.0 * x0;
    let p3 = x3 - 3.0 * x2 + 3.0 * x1 - x0;
    (p0, p1, p2, p3)
}

impl From<CubicBez> for PathSeg {
    fn from(cubic_bez: CubicBez) -> PathSeg {
        PathSeg::Cubic(cubic_bez)
    }
}

impl From<Line> for PathSeg {
    fn from(line: Line) -> PathSeg {
        PathSeg::Line(line)
    }
}

impl From<QuadBez> for PathSeg {
    fn from(quad_bez: QuadBez) -> PathSeg {
        PathSeg::Quad(quad_bez)
    }
}

impl Shape for BezPath {
    type PathElementsIter = std::vec::IntoIter<PathEl>;

    fn path_elements(&self, _tolerance: f64) -> Self::PathElementsIter {
        self.0.clone().into_iter()
    }

    fn to_path(&self, _tolerance: f64) -> BezPath {
        self.clone()
    }

    fn into_path(self, _tolerance: f64) -> BezPath {
        self
    }

    /// Signed area.
    fn area(&self) -> f64 {
        self.elements().area()
    }

    fn perimeter(&self, accuracy: f64) -> f64 {
        self.elements().perimeter(accuracy)
    }

    /// Winding number of point.
    fn winding(&self, pt: Point) -> i32 {
        self.elements().winding(pt)
    }

    fn bounding_box(&self) -> Rect {
        self.elements().bounding_box()
    }

    fn as_path_slice(&self) -> Option<&[PathEl]> {
        Some(&self.0)
    }
}

impl PathEl {
    /// Is this path element finite?
    #[inline]
    pub fn is_finite(&self) -> bool {
        match self {
            PathEl::MoveTo(p) => p.is_finite(),
            PathEl::LineTo(p) => p.is_finite(),
            PathEl::QuadTo(p, p2) => p.is_finite() && p2.is_finite(),
            PathEl::CurveTo(p, p2, p3) => p.is_finite() && p2.is_finite() && p3.is_finite(),
            PathEl::ClosePath => true,
        }
    }

    /// Is this path element NaN?
    #[inline]
    pub fn is_nan(&self) -> bool {
        match self {
            PathEl::MoveTo(p) => p.is_nan(),
            PathEl::LineTo(p) => p.is_nan(),
            PathEl::QuadTo(p, p2) => p.is_nan() || p2.is_nan(),
            PathEl::CurveTo(p, p2, p3) => p.is_nan() || p2.is_nan() || p3.is_nan(),
            PathEl::ClosePath => false,
        }
    }

    /// Returns the end point for the element if it can be determined
    #[inline]
    pub fn end(&self) -> Option<Point> {
        match self {
            PathEl::MoveTo(p) => Some(*p),
            PathEl::LineTo(p) => Some(*p),
            PathEl::QuadTo(_, p) => Some(*p),
            PathEl::CurveTo(_, _, p) => Some(*p),
            PathEl::ClosePath => None,
        }
    }
}

/// Implements [`Shape`] for a slice of [`PathEl`], provided that the first element of the slice is
/// not a `PathEl::ClosePath`. If it is, several of these functions will panic.
///
/// If the slice starts with `LineTo`, `QuadTo`, or `CurveTo`, it will be treated as a `MoveTo`.
impl<'a> Shape for &'a [PathEl] {
    type PathElementsIter = std::iter::Cloned<std::slice::Iter<'a, PathEl>>;

    #[inline]
    fn path_elements(&self, _tolerance: f64) -> Self::PathElementsIter {
        self.iter().cloned()
    }

    fn to_path(&self, _tolerance: f64) -> BezPath {
        BezPath::from_vec(self.to_vec())
    }

    /// Signed area.
    fn area(&self) -> f64 {
        segments(self.iter().copied()).area()
    }

    fn perimeter(&self, accuracy: f64) -> f64 {
        segments(self.iter().copied()).perimeter(accuracy)
    }

    /// Winding number of point.
    fn winding(&self, pt: Point) -> i32 {
        segments(self.iter().copied()).winding(pt)
    }

    fn bounding_box(&self) -> Rect {
        segments(self.iter().copied()).bounding_box()
    }

    #[inline]
    fn as_path_slice(&self) -> Option<&[PathEl]> {
        Some(self)
    }
}

/// An iterator for path segments.
pub struct PathSegIter {
    seg: PathSeg,
    ix: usize,
}

impl Shape for PathSeg {
    type PathElementsIter = PathSegIter;

    #[inline]
    fn path_elements(&self, _tolerance: f64) -> PathSegIter {
        PathSegIter { seg: *self, ix: 0 }
    }

    /// The area under the curve.
    ///
    /// We could just return 0, but this seems more useful.
    fn area(&self) -> f64 {
        self.signed_area()
    }

    #[inline]
    fn perimeter(&self, accuracy: f64) -> f64 {
        self.arclen(accuracy)
    }

    fn winding(&self, _pt: Point) -> i32 {
        0
    }

    #[inline]
    fn bounding_box(&self) -> Rect {
        ParamCurveExtrema::bounding_box(self)
    }

    fn as_line(&self) -> Option<Line> {
        if let PathSeg::Line(line) = self {
            Some(*line)
        } else {
            None
        }
    }
}

impl Iterator for PathSegIter {
    type Item = PathEl;

    fn next(&mut self) -> Option<PathEl> {
        self.ix += 1;
        match (self.ix, self.seg) {
            // yes I could do some fancy bindings thing here but... :shrug:
            (1, PathSeg::Line(seg)) => Some(PathEl::MoveTo(seg.p0)),
            (1, PathSeg::Quad(seg)) => Some(PathEl::MoveTo(seg.p0)),
            (1, PathSeg::Cubic(seg)) => Some(PathEl::MoveTo(seg.p0)),
            (2, PathSeg::Line(seg)) => Some(PathEl::LineTo(seg.p1)),
            (2, PathSeg::Quad(seg)) => Some(PathEl::QuadTo(seg.p1, seg.p2)),
            (2, PathSeg::Cubic(seg)) => Some(PathEl::CurveTo(seg.p1, seg.p2, seg.p3)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Circle, DEFAULT_ACCURACY};

    use super::*;

    fn assert_approx_eq(x: f64, y: f64) {
        assert!((x - y).abs() < 1e-8, "{} != {}", x, y);
    }

    #[test]
    #[should_panic(expected = "no open subpath")]
    #[cfg(debug_assertions)] // Only provides proper panic message in debug mode
    fn test_elements_to_segments_starts_on_closepath() {
        let mut path = BezPath::new();
        path.close_path();
        path.segments().next();
    }

    #[test]
    fn test_elements_to_segments_closepath_refers_to_last_moveto() {
        let mut path = BezPath::new();
        path.move_to((5.0, 5.0));
        path.line_to((15.0, 15.0));
        path.move_to((10.0, 10.0));
        path.line_to((15.0, 15.0));
        path.close_path();
        assert_eq!(
            path.segments().collect::<Vec<_>>().last(),
            Some(&Line::new((15.0, 15.0), (10.0, 10.0)).into()),
        );
    }

    #[test]
    #[should_panic(expected = "no open subpath")]
    #[cfg(debug_assertions)] // Only provides proper panic message in debug mode
    fn test_must_not_start_on_quad() {
        let mut path = BezPath::new();
        path.quad_to((5.0, 5.0), (10.0, 10.0));
        path.line_to((15.0, 15.0));
        path.close_path();
    }

    #[test]
    fn test_split_at_moves() {
        let mut path = BezPath::new();
        path.move_to((0.0, 0.0));
        path.line_to((10.0, 0.0));
        path.move_to((0.0, 1.0));
        path.line_to((10.0, 1.0));
        path.line_to((20.0, 1.0));
        path.move_to((0.0, 2.0));
        path.move_to((0.0, 3.0));
        path.line_to((10.0, 3.0));
        path.line_to((20.0, 3.0));
        path.line_to((30.0, 3.0));

        let path_list = path.split_at_moves();

        assert_eq!(path_list.len(), 3);
        assert_eq!(path_list[0].0.len(), 2);
        assert_eq!(path_list[1].0.len(), 3);
        assert_eq!(path_list[2].0.len(), 4);
    }

    #[test]
    fn test_close_subpaths() {
        let mut path = BezPath::new();
        path.move_to((0.0, 0.0));
        path.line_to((10.0, 0.0));
        path.move_to((0.0, 100.0));
        path.line_to((10.0, 100.0));
        path.line_to((20.0, 100.0));
        path.move_to((0.0, 200.0));
        path.move_to((0.0, 300.0));
        path.line_to((10.0, 300.0));
        path.line_to((20.0, 300.0));
        path.line_to((30.0, 300.0));
        path.close_subpaths();

        assert_eq!(path.0.len(), 13);
    }

    #[test]
    fn test_break_at_self_intersections() {
        let mut path = BezPath::new();
        path.move_to((0.0, 0.0));
        path.line_to((1.0, 1.0));
        path.line_to((0.0, 1.0));
        path.line_to((1.0, 1.0));

        let intersections = vec![((1, 0.5), (3, 0.5))];

        let mut path2 = path.clone();
        BezPath::break_at_self_intersections(&mut path2, &intersections);
        assert_eq!(path2.0.len(), path.0.len() + intersections.len() * 4);
    }

    #[test]
    fn test_break_at_intersections() {
        let mut path = BezPath::new();
        path.move_to((0.0, 0.0));
        path.line_to((1.0, 0.0));
        path.line_to((1.0, 1.0));
        path.line_to((0.0, 1.0));
        path.close_path();

        let mut path1 = path.clone();
        let mut path2 = Affine::translate((0.5, 0.5)) * path.clone();

        let intersections = vec![((2, 0.5), (1, 0.5)), ((3, 0.5), (4, 0.5))];
        BezPath::break_at_intersections(&mut path1, &mut path2, &intersections);
        assert_eq!(path1.0.len(), path.0.len() + intersections.len() * 2);
        assert_eq!(path2.0.len(), path1.0.len());
    }

    #[test]
    fn test_intersect_line() {
        let h_line = Line::new((0.0, 0.0), (100.0, 0.0));
        let v_line = Line::new((10.0, -10.0), (10.0, 10.0));
        let intersection = PathSeg::Line(h_line).intersect_line(v_line)[0];
        assert_approx_eq(intersection.segment_t, 0.1);
        assert_approx_eq(intersection.line_t, 0.5);

        let v_line = Line::new((-10.0, -10.0), (-10.0, 10.0));
        assert!(PathSeg::Line(h_line).intersect_line(v_line).is_empty());

        let v_line = Line::new((10.0, 10.0), (10.0, 20.0));
        assert!(PathSeg::Line(h_line).intersect_line(v_line).is_empty());
    }

    #[test]
    fn test_intersect_qad() {
        let q = QuadBez::new((0.0, -10.0), (10.0, 20.0), (20.0, -10.0));
        let v_line = Line::new((10.0, -10.0), (10.0, 10.0));
        assert_eq!(PathSeg::Quad(q).intersect_line(v_line).len(), 1);
        let intersection = PathSeg::Quad(q).intersect_line(v_line)[0];
        assert_approx_eq(intersection.segment_t, 0.5);
        assert_approx_eq(intersection.line_t, 0.75);

        let h_line = Line::new((0.0, 0.0), (100.0, 0.0));
        assert_eq!(PathSeg::Quad(q).intersect_line(h_line).len(), 2);
    }

    #[test]
    fn test_intersect_cubic() {
        let c = CubicBez::new((0.0, -10.0), (10.0, 20.0), (20.0, -20.0), (30.0, 10.0));
        let v_line = Line::new((10.0, -10.0), (10.0, 10.0));
        assert_eq!(PathSeg::Cubic(c).intersect_line(v_line).len(), 1);
        let intersection = PathSeg::Cubic(c).intersect_line(v_line)[0];
        assert_approx_eq(intersection.segment_t, 0.333333333);
        assert_approx_eq(intersection.line_t, 0.592592592);

        let h_line = Line::new((0.0, 0.0), (100.0, 0.0));
        assert_eq!(PathSeg::Cubic(c).intersect_line(h_line).len(), 3);
    }

    #[test]
    fn test_contains() {
        let mut path = BezPath::new();
        path.move_to((0.0, 0.0));
        path.line_to((1.0, 1.0));
        path.line_to((2.0, 0.0));
        path.close_path();
        assert_eq!(path.winding(Point::new(1.0, 0.5)), -1);
        assert!(path.contains(Point::new(1.0, 0.5)));
    }

    // get_seg(i) should produce the same results as path_segments().nth(i - 1).
    #[test]
    fn test_get_seg() {
        let circle = Circle::new((10.0, 10.0), 2.0).to_path(DEFAULT_ACCURACY);
        let segments = circle.path_segments(DEFAULT_ACCURACY).collect::<Vec<_>>();
        let get_segs = (1..usize::MAX)
            .map_while(|i| circle.get_seg(i))
            .collect::<Vec<_>>();
        assert_eq!(segments, get_segs);
    }

    #[test]
    fn test_self_intersections() {
        let mut path = BezPath::new();
        path.move_to((0., 0.));
        path.line_to((0., 1.));
        path.curve_to((2., 0.5), (-1., 0.5), (1., 1.));
        path.line_to((-0.5, 0.));
        path.close_path();

        let intersects = path.self_intersections(0.);

        let mut path2 = path.clone();
        BezPath::break_at_self_intersections(&mut path2, &intersects);

        assert_eq!(intersects.len(), 4);
        assert_eq!(
            path2.elements().len(),
            path.elements().len() + intersects.len() * 4
        );
    }
}
