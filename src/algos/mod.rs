//! Assignment solvers for the bipartite matching problem.
//!
//! This module provides matching problem solver implementation with a common interface.

use std::time::Duration;
use crate::algos::auction::auction;
use crate::algos::greedy::greedy;
use crate::algos::hungarian::hungarian;
use crate::math::math::negate_mat;

pub mod hungarian;
pub mod auction;
pub mod greedy;

/// Matching problem solution
#[derive(Debug, Clone)]
pub struct AssignmentResult {
    /// Assignment mapping, index is job idx, value is assigned job idx.
    pub assignment: Vec<usize>,
    /// Runtime of the assignment.
    pub runtime: Duration
}

/// Implementations mush solve the bipartite matching problem, assigning workers to jobs while
/// optimizing total cost.
pub trait AssignmentAlgorithm {
    /// Solve the assignment problem.
    fn solve(&self, cost_matrix: &Vec<Vec<f64>>) -> Option<AssignmentResult>;
}

pub struct Auction {
    pub eps: f64,
}
pub struct Greedy;
pub struct Hungarian;

impl AssignmentAlgorithm for Auction {
    fn solve(&self, cost_mat: &Vec<Vec<f64>>) -> Option<AssignmentResult> {
        auction(cost_mat, self.eps)
    }
}

impl AssignmentAlgorithm for Greedy {
    fn solve(&self, cost_mat: &Vec<Vec<f64>>) -> Option<AssignmentResult> {
        greedy(cost_mat)
    }
}

impl AssignmentAlgorithm for Hungarian {
    fn solve(&self, cost_mat: &Vec<Vec<f64>>) -> Option<AssignmentResult> {
        let negated = negate_mat(cost_mat);
        hungarian(&negated)
    }
}

pub enum Solver {
    Auction(Auction),
    Greedy(Greedy),
    Hungarian(Hungarian),
}

impl Solver {
    pub fn solve(&self, cost_mat: &Vec<Vec<f64>>) -> Option<AssignmentResult> {
        match self {
            Solver::Auction(auction) => auction.solve(cost_mat),
            Solver::Greedy(greedy) => greedy.solve(cost_mat),
            Solver::Hungarian(hungarian) => hungarian.solve(cost_mat)
        }
    }
}